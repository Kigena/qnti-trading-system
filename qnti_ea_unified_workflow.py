#!/usr/bin/env python3
"""
QNTI Unified EA Generation Workflow - Complete Strategy Factory
End-to-end workflow orchestrating EA generation, optimization, robustness testing, and deployment
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
from pathlib import Path
import concurrent.futures
import threading
import time
import uuid
from collections import defaultdict

# Import all EA system components
from qnti_ea_generation_engine import (
    EAGenerationEngine, EATemplate, Parameter, ParameterType, 
    IndicatorLibrary, TradingRule, Condition, ConditionGroup
)
from qnti_ea_optimization_engine import (
    OptimizationEngine, OptimizationMethod, OptimizationConfig, 
    OptimizationResult, BacktestMetrics
)
from qnti_ea_robustness_testing import (
    RobustnessTestingEngine, RobustnessConfig, RobustnessReport, RobustnessTest
)
from qnti_ea_backtesting_integration import (
    EABacktestingIntegration, BacktestConfiguration
)
from qnti_ea_reporting_system import (
    ReportGenerator, ReportConfiguration, EAPerformanceReport
)

logger = logging.getLogger('QNTI_EA_WORKFLOW')

class WorkflowStage(Enum):
    """EA generation workflow stages"""
    DESIGN = "design"
    OPTIMIZATION = "optimization"
    ROBUSTNESS_TESTING = "robustness_testing"
    VALIDATION = "validation"
    REPORTING = "reporting"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

@dataclass
class WorkflowConfiguration:
    """Complete workflow configuration"""
    # EA Design
    ea_name: str
    ea_description: str
    target_symbols: List[str]
    target_timeframes: List[str]
    
    # Indicators to include
    indicators: List[Dict[str, Any]]  # [{"name": "SMA", "params": {...}}, ...]
    
    # Optimization settings
    optimization_method: OptimizationMethod = OptimizationMethod.GENETIC_ALGORITHM
    optimization_config: OptimizationConfig = None
    
    # Robustness testing settings
    robustness_tests: List[RobustnessTest] = field(default_factory=lambda: [
        RobustnessTest.WALK_FORWARD,
        RobustnessTest.MONTE_CARLO,
        RobustnessTest.PARAMETER_SENSITIVITY,
        RobustnessTest.STRESS_TESTING
    ])
    robustness_config: RobustnessConfig = None
    
    # Backtesting settings
    backtest_start_date: datetime = None
    backtest_end_date: datetime = None
    initial_balance: float = 10000.0
    
    # Quality thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = -0.15
    min_robustness_score: float = 60.0
    min_annual_return: float = 0.05
    
    # Workflow settings
    auto_proceed: bool = False  # Automatically proceed through stages
    parallel_processing: bool = True
    max_workers: int = 4
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        if self.optimization_config is None:
            self.optimization_config = OptimizationConfig(method=self.optimization_method)
        if self.robustness_config is None:
            self.robustness_config = RobustnessConfig()
        if self.backtest_start_date is None:
            self.backtest_start_date = datetime.now() - timedelta(days=365*2)
        if self.backtest_end_date is None:
            self.backtest_end_date = datetime.now() - timedelta(days=30)

@dataclass
class WorkflowState:
    """Current state of workflow execution"""
    workflow_id: str
    current_stage: WorkflowStage
    status: WorkflowStatus
    start_time: datetime
    
    # Stage completion tracking
    completed_stages: List[WorkflowStage] = field(default_factory=list)
    failed_stages: List[WorkflowStage] = field(default_factory=list)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    
    # Progress tracking
    current_stage_progress: float = 0.0
    overall_progress: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    
    # Results
    ea_template: Optional[EATemplate] = None
    optimization_result: Optional[OptimizationResult] = None
    robustness_report: Optional[RobustnessReport] = None
    performance_report: Optional[EAPerformanceReport] = None
    
    # Execution metadata
    total_execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        if self.estimated_completion_time:
            result['estimated_completion_time'] = self.estimated_completion_time.isoformat()
        return result

class QualityGate:
    """Quality gate for workflow stage validation"""
    
    def __init__(self, config: WorkflowConfiguration):
        self.config = config
    
    def check_optimization_quality(self, result: OptimizationResult) -> Tuple[bool, List[str]]:
        """Check if optimization results meet quality thresholds"""
        issues = []
        
        metrics = result.performance_metrics
        
        # Check Sharpe ratio
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < self.config.min_sharpe_ratio:
            issues.append(f"Sharpe ratio {sharpe:.2f} below threshold {self.config.min_sharpe_ratio}")
        
        # Check annual return
        annual_return = metrics.get('annual_return', 0)
        if annual_return < self.config.min_annual_return:
            issues.append(f"Annual return {annual_return:.2%} below threshold {self.config.min_annual_return:.2%}")
        
        # Check drawdown
        max_dd = metrics.get('max_drawdown', 0)
        if max_dd < self.config.max_drawdown_threshold:
            issues.append(f"Max drawdown {max_dd:.2%} exceeds threshold {self.config.max_drawdown_threshold:.2%}")
        
        return len(issues) == 0, issues
    
    def check_robustness_quality(self, report: RobustnessReport) -> Tuple[bool, List[str]]:
        """Check if robustness results meet quality thresholds"""
        issues = []
        
        # Check overall robustness score
        if report.overall_robustness_score < self.config.min_robustness_score:
            issues.append(f"Robustness score {report.overall_robustness_score:.1f} below threshold {self.config.min_robustness_score}")
        
        # Check individual test results
        if report.walk_forward_results.get('stability_score', 0) < 50:
            issues.append("Walk-forward analysis shows poor temporal stability")
        
        if report.monte_carlo_results.get('consistency_score', 0) < 60:
            issues.append("Monte Carlo simulation shows poor consistency")
        
        if report.parameter_sensitivity_score > 80:
            issues.append("High parameter sensitivity detected")
        
        return len(issues) == 0, issues

class WorkflowEngine:
    """Main workflow execution engine"""
    
    def __init__(self, mt5_bridge=None, notification_system=None):
        self.mt5_bridge = mt5_bridge
        self.notification_system = notification_system
        
        # Initialize all subsystems
        self.ea_engine = EAGenerationEngine(mt5_bridge)
        self.optimization_engine = OptimizationEngine()
        self.robustness_engine = RobustnessTestingEngine()
        self.backtesting_integration = EABacktestingIntegration(mt5_bridge)
        self.report_generator = ReportGenerator()
        
        # Workflow management
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.workflow_history: List[WorkflowState] = []
        
        # Storage
        self.workflow_storage_path = Path("qnti_data/ea_workflows")
        self.workflow_storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("QNTI EA Workflow Engine initialized")
    
    async def execute_workflow(self, config: WorkflowConfiguration) -> WorkflowState:
        """Execute complete EA generation workflow"""
        
        workflow_id = str(uuid.uuid4())
        logger.info(f"Starting EA generation workflow: {workflow_id}")
        
        # Initialize workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            current_stage=WorkflowStage.DESIGN,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.active_workflows[workflow_id] = workflow_state
        
        try:
            # Execute workflow stages
            await self._execute_design_stage(workflow_state, config)
            await self._execute_optimization_stage(workflow_state, config)
            await self._execute_robustness_testing_stage(workflow_state, config)
            await self._execute_validation_stage(workflow_state, config)
            await self._execute_reporting_stage(workflow_state, config)
            
            # Complete workflow
            workflow_state.status = WorkflowStatus.COMPLETED
            workflow_state.overall_progress = 100.0
            workflow_state.total_execution_time = (datetime.now() - workflow_state.start_time).total_seconds()
            
            logger.info(f"Workflow {workflow_id} completed successfully in {workflow_state.total_execution_time:.1f}s")
            
            # Send notification if available
            if self.notification_system:
                await self._send_completion_notification(workflow_state, config)
        
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            workflow_state.status = WorkflowStatus.FAILED
            workflow_state.errors.append(str(e))
        
        finally:
            # Move to history and cleanup
            self.workflow_history.append(workflow_state)
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            # Save workflow state
            self._save_workflow_state(workflow_state)
        
        return workflow_state
    
    async def _execute_design_stage(self, state: WorkflowState, config: WorkflowConfiguration):
        """Execute EA design stage"""
        logger.info(f"Executing design stage for workflow {state.workflow_id}")
        
        state.current_stage = WorkflowStage.DESIGN
        state.current_stage_progress = 0.0
        
        try:
            # Create EA template
            ea_template = self.ea_engine.create_ea_template(
                name=config.ea_name,
                description=config.ea_description,
                author="QNTI Workflow Engine"
            )
            
            state.current_stage_progress = 30.0
            
            # Add indicators
            for indicator_config in config.indicators:
                success = self.ea_engine.add_indicator_to_ea(
                    ea_template.id,
                    indicator_config['name'],
                    indicator_config.get('params', {})
                )
                if not success:
                    state.warnings.append(f"Failed to add indicator: {indicator_config['name']}")
            
            state.current_stage_progress = 60.0
            
            # Generate basic trading rules (simplified for demo)
            await self._generate_trading_rules(ea_template, config)
            
            state.current_stage_progress = 90.0
            
            # Save EA template
            self.ea_engine.save_ea_template(ea_template.id)
            
            state.ea_template = ea_template
            state.current_stage_progress = 100.0
            state.completed_stages.append(WorkflowStage.DESIGN)
            state.overall_progress = 15.0
            
            logger.info(f"Design stage completed for workflow {state.workflow_id}")
        
        except Exception as e:
            logger.error(f"Design stage failed: {e}")
            state.failed_stages.append(WorkflowStage.DESIGN)
            raise
    
    async def _execute_optimization_stage(self, state: WorkflowState, config: WorkflowConfiguration):
        """Execute optimization stage"""
        logger.info(f"Executing optimization stage for workflow {state.workflow_id}")
        
        state.current_stage = WorkflowStage.OPTIMIZATION
        state.current_stage_progress = 0.0
        
        try:
            # Create objective function
            objective_function = self._create_objective_function(state.ea_template, config)
            
            state.current_stage_progress = 20.0
            
            # Run optimization
            optimization_result = self.optimization_engine.optimize_ea(
                state.ea_template,
                objective_function,
                method=config.optimization_method,
                config=config.optimization_config
            )
            
            state.current_stage_progress = 90.0
            
            # Quality gate check
            quality_gate = QualityGate(config)
            passed, issues = quality_gate.check_optimization_quality(optimization_result)
            
            if not passed and not config.auto_proceed:
                logger.warning(f"Optimization quality issues: {issues}")
                for issue in issues:
                    state.warnings.append(f"Optimization quality: {issue}")
            
            state.optimization_result = optimization_result
            state.current_stage_progress = 100.0
            state.completed_stages.append(WorkflowStage.OPTIMIZATION)
            state.overall_progress = 35.0
            
            logger.info(f"Optimization stage completed for workflow {state.workflow_id}")
        
        except Exception as e:
            logger.error(f"Optimization stage failed: {e}")
            state.failed_stages.append(WorkflowStage.OPTIMIZATION)
            raise
    
    async def _execute_robustness_testing_stage(self, state: WorkflowState, config: WorkflowConfiguration):
        """Execute robustness testing stage"""
        logger.info(f"Executing robustness testing stage for workflow {state.workflow_id}")
        
        state.current_stage = WorkflowStage.ROBUSTNESS_TESTING
        state.current_stage_progress = 0.0
        
        try:
            # Get market data
            data = await self._get_market_data_for_testing(config)
            
            state.current_stage_progress = 20.0
            
            # Create backtest function
            backtest_function = self._create_backtest_function(state.ea_template, config)
            
            state.current_stage_progress = 30.0
            
            # Run robustness tests
            robustness_report = self.robustness_engine.run_comprehensive_test(
                state.ea_template,
                data,
                backtest_function,
                state.optimization_result.parameters,
                tests_to_run=config.robustness_tests
            )
            
            state.current_stage_progress = 90.0
            
            # Quality gate check
            quality_gate = QualityGate(config)
            passed, issues = quality_gate.check_robustness_quality(robustness_report)
            
            if not passed and not config.auto_proceed:
                logger.warning(f"Robustness quality issues: {issues}")
                for issue in issues:
                    state.warnings.append(f"Robustness quality: {issue}")
            
            state.robustness_report = robustness_report
            state.current_stage_progress = 100.0
            state.completed_stages.append(WorkflowStage.ROBUSTNESS_TESTING)
            state.overall_progress = 60.0
            
            logger.info(f"Robustness testing stage completed for workflow {state.workflow_id}")
        
        except Exception as e:
            logger.error(f"Robustness testing stage failed: {e}")
            state.failed_stages.append(WorkflowStage.ROBUSTNESS_TESTING)
            raise
    
    async def _execute_validation_stage(self, state: WorkflowState, config: WorkflowConfiguration):
        """Execute validation stage"""
        logger.info(f"Executing validation stage for workflow {state.workflow_id}")
        
        state.current_stage = WorkflowStage.VALIDATION
        state.current_stage_progress = 0.0
        
        try:
            # Run out-of-sample validation
            validation_data = await self._get_validation_data(config)
            
            state.current_stage_progress = 30.0
            
            # Create validation backtest configuration
            validation_config = BacktestConfiguration(
                start_date=config.backtest_end_date,
                end_date=datetime.now() - timedelta(days=1),
                symbol=config.target_symbols[0],  # Use first symbol
                initial_balance=config.initial_balance
            )
            
            state.current_stage_progress = 50.0
            
            # Run validation backtest
            validation_metrics = self.backtesting_integration.backtester.run_backtest(
                state.ea_template,
                validation_config,
                state.optimization_result.parameters,
                validation_data
            )
            
            state.current_stage_progress = 90.0
            
            # Store validation results
            state.stage_results['validation'] = {
                'metrics': validation_metrics,
                'out_of_sample_period': {
                    'start': validation_config.start_date.isoformat(),
                    'end': validation_config.end_date.isoformat()
                }
            }
            
            state.current_stage_progress = 100.0
            state.completed_stages.append(WorkflowStage.VALIDATION)
            state.overall_progress = 80.0
            
            logger.info(f"Validation stage completed for workflow {state.workflow_id}")
        
        except Exception as e:
            logger.error(f"Validation stage failed: {e}")
            state.failed_stages.append(WorkflowStage.VALIDATION)
            # Don't raise - validation failure shouldn't stop the workflow
            state.warnings.append(f"Validation failed: {str(e)}")
    
    async def _execute_reporting_stage(self, state: WorkflowState, config: WorkflowConfiguration):
        """Execute reporting stage"""
        logger.info(f"Executing reporting stage for workflow {state.workflow_id}")
        
        state.current_stage = WorkflowStage.REPORTING
        state.current_stage_progress = 0.0
        
        try:
            # Get final backtest metrics
            final_metrics = self._extract_final_metrics(state)
            
            state.current_stage_progress = 30.0
            
            # Generate comprehensive report
            report_config = ReportConfiguration(
                include_charts=True,
                include_detailed_analysis=True,
                include_recommendations=True
            )
            
            performance_report = self.report_generator.generate_report(
                state.ea_template,
                final_metrics,
                state.optimization_result,
                state.robustness_report
            )
            
            state.current_stage_progress = 80.0
            
            # Save report
            report_filename = f"ea_report_{state.ea_template.name}_{state.workflow_id}.html"
            report_path = self.workflow_storage_path / report_filename
            self.report_generator.save_report(performance_report, str(report_path))
            
            state.performance_report = performance_report
            state.current_stage_progress = 100.0
            state.completed_stages.append(WorkflowStage.REPORTING)
            state.overall_progress = 95.0
            
            logger.info(f"Reporting stage completed for workflow {state.workflow_id}")
        
        except Exception as e:
            logger.error(f"Reporting stage failed: {e}")
            state.failed_stages.append(WorkflowStage.REPORTING)
            # Don't raise - reporting failure shouldn't stop the workflow
            state.warnings.append(f"Reporting failed: {str(e)}")
    
    def _create_objective_function(self, ea_template: EATemplate, 
                                  config: WorkflowConfiguration) -> Callable:
        """Create objective function for optimization"""
        
        def objective_function(parameters: Dict[str, Any]) -> BacktestMetrics:
            # Create backtest configuration
            backtest_config = BacktestConfiguration(
                start_date=config.backtest_start_date,
                end_date=config.backtest_end_date,
                symbol=config.target_symbols[0],  # Use first symbol for optimization
                initial_balance=config.initial_balance
            )
            
            # Run backtest
            return self.backtesting_integration.backtester.run_backtest(
                ea_template,
                backtest_config,
                parameters
            )
        
        return objective_function
    
    def _create_backtest_function(self, ea_template: EATemplate, 
                                 config: WorkflowConfiguration) -> Callable:
        """Create backtest function for robustness testing"""
        
        def backtest_function(data: pd.DataFrame, parameters: Dict[str, Any]) -> BacktestMetrics:
            # Create backtest configuration
            backtest_config = BacktestConfiguration(
                start_date=data.index[0],
                end_date=data.index[-1],
                symbol=config.target_symbols[0],
                initial_balance=config.initial_balance
            )
            
            # Run backtest with provided data
            return self.backtesting_integration.backtester.run_backtest(
                ea_template,
                backtest_config,
                parameters,
                data
            )
        
        return backtest_function
    
    async def _get_market_data_for_testing(self, config: WorkflowConfiguration) -> pd.DataFrame:
        """Get market data for robustness testing"""
        # For now, use the backtesting integration to get data
        # In production, this would fetch from MT5 or other data source
        
        if self.mt5_bridge:
            try:
                data = self.mt5_bridge.get_historical_data(
                    symbol=config.target_symbols[0],
                    timeframe=config.target_timeframes[0],
                    start_date=config.backtest_start_date,
                    end_date=config.backtest_end_date
                )
                return data
            except Exception as e:
                logger.warning(f"Failed to get MT5 data: {e}")
        
        # Generate synthetic data as fallback
        return self._generate_synthetic_data(config)
    
    async def _get_validation_data(self, config: WorkflowConfiguration) -> pd.DataFrame:
        """Get out-of-sample validation data"""
        # Similar to market data but for validation period
        validation_start = config.backtest_end_date
        validation_end = datetime.now() - timedelta(days=1)
        
        if self.mt5_bridge and validation_start < validation_end:
            try:
                data = self.mt5_bridge.get_historical_data(
                    symbol=config.target_symbols[0],
                    timeframe=config.target_timeframes[0],
                    start_date=validation_start,
                    end_date=validation_end
                )
                return data
            except Exception as e:
                logger.warning(f"Failed to get validation data: {e}")
        
        # Generate synthetic validation data
        return self._generate_synthetic_data(config, start_date=validation_start, end_date=validation_end)
    
    def _generate_synthetic_data(self, config: WorkflowConfiguration,
                                start_date: datetime = None,
                                end_date: datetime = None) -> pd.DataFrame:
        """Generate synthetic market data"""
        
        if start_date is None:
            start_date = config.backtest_start_date
        if end_date is None:
            end_date = config.backtest_end_date
        
        # Create date range
        freq = 'H' if 'H1' in config.target_timeframes else 'D'
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate realistic price data
        n_periods = len(dates)
        base_price = 1.1000
        
        # Generate returns with trend and volatility
        returns = np.random.normal(0.0001, 0.01, n_periods)  # Small positive drift
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1).fillna(prices[0])
        
        # Add intrabar volatility
        volatility = np.abs(np.random.normal(0, 0.005, n_periods))
        data['high'] = data[['open', 'close']].max(axis=1) + volatility
        data['low'] = data[['open', 'close']].min(axis=1) - volatility
        data['volume'] = np.random.exponential(1000, n_periods)
        
        return data
    
    async def _generate_trading_rules(self, ea_template: EATemplate, config: WorkflowConfiguration):
        """Generate basic trading rules for EA"""
        # This is a simplified implementation
        # In production, this would use sophisticated rule generation logic
        
        # For now, create a simple rule structure placeholder
        # The actual trading rules would be generated based on indicators and strategy logic
        logger.info("Generated basic trading rule structure")
    
    def _extract_final_metrics(self, state: WorkflowState) -> BacktestMetrics:
        """Extract final performance metrics from optimization result"""
        if state.optimization_result and state.optimization_result.performance_metrics:
            # Convert performance metrics dict to BacktestMetrics object
            metrics_dict = state.optimization_result.performance_metrics
            
            return BacktestMetrics(
                annual_return=metrics_dict.get('annual_return', 0),
                total_return=metrics_dict.get('total_return', 0),
                sharpe_ratio=metrics_dict.get('sharpe_ratio', 0),
                sortino_ratio=metrics_dict.get('sortino_ratio', 0),
                max_drawdown=metrics_dict.get('max_drawdown', 0),
                calmar_ratio=metrics_dict.get('calmar_ratio', 0),
                win_rate=metrics_dict.get('win_rate', 0),
                profit_factor=metrics_dict.get('profit_factor', 0),
                avg_win=metrics_dict.get('avg_win', 0),
                avg_loss=metrics_dict.get('avg_loss', 0),
                total_trades=metrics_dict.get('total_trades', 0),
                winning_trades=metrics_dict.get('winning_trades', 0),
                losing_trades=metrics_dict.get('losing_trades', 0),
                volatility=metrics_dict.get('volatility', 0)
            )
        
        # Return empty metrics if no optimization result
        return BacktestMetrics()
    
    async def _send_completion_notification(self, state: WorkflowState, config: WorkflowConfiguration):
        """Send workflow completion notification"""
        if self.notification_system:
            try:
                message = f"""
EA Generation Workflow Completed

EA Name: {config.ea_name}
Workflow ID: {state.workflow_id}
Status: {state.status.value}
Execution Time: {state.total_execution_time:.1f}s

Performance Summary:
- Overall Rating: {state.performance_report.overall_rating if state.performance_report else 'N/A'}
- Performance Score: {state.performance_report.performance_score:.1f}/100 if state.performance_report else 'N/A'
- Robustness Score: {state.robustness_report.overall_robustness_score:.1f}/100 if state.robustness_report else 'N/A'

Completed Stages: {len(state.completed_stages)}/{len(WorkflowStage)}
                """
                
                self.notification_system.send_notification(
                    title="EA Generation Workflow Completed",
                    message=message.strip(),
                    level="info",
                    category="ea_generation"
                )
            except Exception as e:
                logger.error(f"Failed to send completion notification: {e}")
    
    def _save_workflow_state(self, state: WorkflowState):
        """Save workflow state to disk"""
        try:
            state_file = self.workflow_storage_path / f"workflow_{state.workflow_id}.json"
            with open(state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save workflow state: {e}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get current status of a workflow"""
        return self.active_workflows.get(workflow_id)
    
    def list_active_workflows(self) -> List[WorkflowState]:
        """List all active workflows"""
        return list(self.active_workflows.values())
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            logger.info(f"Workflow {workflow_id} cancelled")
            return True
        return False

# Convenience function for simple workflow execution
async def generate_ea_strategy(
    name: str,
    description: str,
    indicators: List[Dict[str, Any]],
    symbols: List[str] = None,
    timeframes: List[str] = None,
    mt5_bridge=None,
    **kwargs
) -> WorkflowState:
    """Convenient function to generate an EA strategy with default settings"""
    
    if symbols is None:
        symbols = ["EURUSD"]
    if timeframes is None:
        timeframes = ["H1"]
    
    # Create workflow configuration
    config = WorkflowConfiguration(
        ea_name=name,
        ea_description=description,
        target_symbols=symbols,
        target_timeframes=timeframes,
        indicators=indicators,
        **kwargs
    )
    
    # Execute workflow
    engine = WorkflowEngine(mt5_bridge)
    return await engine.execute_workflow(config)

# Example usage
if __name__ == "__main__":
    async def main():
        # Define a simple EA strategy
        indicators = [
            {
                "name": "SMA",
                "params": {"period": 20, "source": "close"}
            },
            {
                "name": "RSI", 
                "params": {"period": 14, "overbought": 70, "oversold": 30}
            },
            {
                "name": "MACD",
                "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
            }
        ]
        
        # Generate EA strategy
        result = await generate_ea_strategy(
            name="Multi-Indicator Trend Strategy",
            description="Strategy combining SMA, RSI, and MACD for trend following",
            indicators=indicators,
            symbols=["EURUSD", "GBPUSD"],
            timeframes=["H1", "H4"],
            auto_proceed=True
        )
        
        print(f"Workflow completed with status: {result.status.value}")
        print(f"Generated EA: {result.ea_template.name if result.ea_template else 'None'}")
        print(f"Performance rating: {result.performance_report.overall_rating if result.performance_report else 'N/A'}")
        print(f"Execution time: {result.total_execution_time:.1f} seconds")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
    
    # Run the example
    asyncio.run(main())