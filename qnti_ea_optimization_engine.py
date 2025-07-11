#!/usr/bin/env python3
"""
QNTI EA Optimization Engine - Advanced Parameter Optimization and Robustness Testing
Multi-algorithm optimization with genetic algorithms, grid search, and machine learning
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle

# Scientific computing imports
try:
    from scipy.optimize import differential_evolution, minimize
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    norm = None
    logging.warning("SciPy not available - some optimization methods disabled")

# Import from main engine
from qnti_ea_generation_engine import (
    EATemplate, Parameter, ParameterType, OptimizationMethod, 
    OptimizationResult, RobustnessTest, RobustnessTestResult
)

logger = logging.getLogger('QNTI_OPTIMIZATION')

@dataclass
class OptimizationConfig:
    """Configuration for optimization process"""
    method: OptimizationMethod
    population_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 5
    convergence_threshold: float = 1e-6
    max_time_seconds: int = 3600
    parallel_workers: int = 4
    random_seed: Optional[int] = None
    
    # Specific algorithm parameters
    grid_resolution: int = 10  # For grid search
    bayesian_iterations: int = 50  # For Bayesian optimization
    pso_inertia: float = 0.7  # For particle swarm
    pso_cognitive: float = 2.0
    pso_social: float = 2.0

@dataclass
class Individual:
    """Individual solution in genetic algorithm"""
    chromosome: Dict[str, Any]
    fitness: float = 0.0
    objectives: Dict[str, float] = field(default_factory=dict)
    age: int = 0
    
    def copy(self):
        """Create a copy of the individual"""
        return Individual(
            chromosome=copy.deepcopy(self.chromosome),
            fitness=self.fitness,
            objectives=copy.deepcopy(self.objectives),
            age=self.age
        )

@dataclass
class BacktestMetrics:
    """Comprehensive backtesting performance metrics"""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    avg_trade_duration: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)
    
    def calculate_composite_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted composite performance score"""
        if weights is None:
            weights = {
                'annual_return': 0.3,
                'sharpe_ratio': 0.25,
                'max_drawdown': -0.2,  # Negative weight (lower is better)
                'win_rate': 0.15,
                'profit_factor': 0.1
            }
        
        score = 0.0
        for metric, weight in weights.items():
            value = getattr(self, metric, 0.0)
            if metric == 'max_drawdown':
                # Convert drawdown to positive score (lower drawdown = higher score)
                value = max(0, 1 - abs(value))
            score += weight * value
        
        return score

class BaseOptimizer(ABC):
    """Base class for optimization algorithms"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.best_individual = None
        self.optimization_history = []
        self.start_time = None
        self.convergence_history = []
        
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
    
    @abstractmethod
    def optimize(self, ea_template: EATemplate, 
                objective_function: Callable) -> OptimizationResult:
        """Run optimization algorithm"""
        pass
    
    def _create_random_individual(self, ea_template: EATemplate) -> Individual:
        """Create random individual within parameter bounds"""
        chromosome = {}
        
        for param in ea_template.parameters:
            if param.param_type == ParameterType.INTEGER:
                value = random.randint(param.min_value, param.max_value)
            elif param.param_type == ParameterType.FLOAT:
                value = random.uniform(param.min_value, param.max_value)
            elif param.param_type == ParameterType.BOOLEAN:
                value = random.choice([True, False])
            elif param.param_type == ParameterType.CHOICE:
                value = random.choice(param.choices)
            else:
                value = param.default_value
            
            chromosome[param.name] = value
        
        return Individual(chromosome=chromosome)
    
    def _evaluate_individual(self, individual: Individual, 
                           objective_function: Callable) -> Individual:
        """Evaluate individual fitness"""
        try:
            metrics = objective_function(individual.chromosome)
            if isinstance(metrics, BacktestMetrics):
                individual.fitness = metrics.calculate_composite_score()
                individual.objectives = metrics.to_dict()
            elif isinstance(metrics, dict):
                individual.objectives = metrics
                individual.fitness = metrics.get('composite_score', 0.0)
            else:
                individual.fitness = float(metrics)
                individual.objectives = {'fitness': individual.fitness}
        except Exception as e:
            logger.error(f"Error evaluating individual: {e}")
            individual.fitness = -1000.0  # Penalty for failed evaluation
        
        return individual
    
    def _is_converged(self) -> bool:
        """Check if optimization has converged"""
        if len(self.convergence_history) < 10:
            return False
        
        recent_scores = self.convergence_history[-10:]
        improvement = max(recent_scores) - min(recent_scores)
        return improvement < self.config.convergence_threshold
    
    def _check_time_limit(self) -> bool:
        """Check if time limit exceeded"""
        if self.start_time is None:
            return False
        
        elapsed = time.time() - self.start_time
        return elapsed > self.config.max_time_seconds

class GeneticOptimizer(BaseOptimizer):
    """Genetic Algorithm optimizer"""
    
    def optimize(self, ea_template: EATemplate, 
                objective_function: Callable) -> OptimizationResult:
        """Run genetic algorithm optimization"""
        self.start_time = time.time()
        logger.info(f"Starting genetic algorithm optimization with {self.config.population_size} individuals")
        
        # Initialize population
        population = []
        for _ in range(self.config.population_size):
            individual = self._create_random_individual(ea_template)
            individual = self._evaluate_individual(individual, objective_function)
            population.append(individual)
        
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = population[0].copy()
        
        generation = 0
        converged = False
        
        while generation < self.config.generations and not converged and not self._check_time_limit():
            # Selection, crossover, and mutation
            new_population = self._evolve_population(population, ea_template, objective_function)
            
            # Update population
            population = new_population
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Update best individual
            if population[0].fitness > self.best_individual.fitness:
                self.best_individual = population[0].copy()
            
            # Track convergence
            avg_fitness = np.mean([ind.fitness for ind in population])
            self.convergence_history.append(avg_fitness)
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {self.best_individual.fitness:.6f}, "
                          f"Avg fitness = {avg_fitness:.6f}")
            
            # Check convergence
            converged = self._is_converged()
            generation += 1
        
        optimization_time = time.time() - self.start_time
        
        result = OptimizationResult(
            ea_id=ea_template.id,
            method=OptimizationMethod.GENETIC_ALGORITHM,
            parameters=self.best_individual.chromosome,
            performance_metrics=self.best_individual.objectives,
            optimization_time=optimization_time,
            iterations=generation,
            convergence_achieved=converged,
            timestamp=datetime.now()
        )
        
        logger.info(f"Genetic algorithm completed: {generation} generations, "
                   f"best fitness = {self.best_individual.fitness:.6f}")
        
        return result
    
    def _evolve_population(self, population: List[Individual], 
                          ea_template: EATemplate, 
                          objective_function: Callable) -> List[Individual]:
        """Evolve population for one generation"""
        new_population = []
        
        # Elitism - keep best individuals
        elite_count = min(self.config.elite_size, len(population))
        new_population.extend([ind.copy() for ind in population[:elite_count]])
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, ea_template)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1, ea_template)
            if random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2, ea_template)
            
            # Evaluate children
            child1 = self._evaluate_individual(child1, objective_function)
            child2 = self._evaluate_individual(child2, objective_function)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        new_population = new_population[:self.config.population_size]
        
        return new_population
    
    def _tournament_selection(self, population: List[Individual], 
                             tournament_size: int = 3) -> Individual:
        """Tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual, 
                  ea_template: EATemplate) -> Tuple[Individual, Individual]:
        """Uniform crossover"""
        child1_chromosome = {}
        child2_chromosome = {}
        
        for param in ea_template.parameters:
            if random.random() < 0.5:
                child1_chromosome[param.name] = parent1.chromosome[param.name]
                child2_chromosome[param.name] = parent2.chromosome[param.name]
            else:
                child1_chromosome[param.name] = parent2.chromosome[param.name]
                child2_chromosome[param.name] = parent1.chromosome[param.name]
        
        child1 = Individual(chromosome=child1_chromosome)
        child2 = Individual(chromosome=child2_chromosome)
        
        return child1, child2
    
    def _mutate(self, individual: Individual, ea_template: EATemplate) -> Individual:
        """Gaussian mutation"""
        mutated_chromosome = copy.deepcopy(individual.chromosome)
        
        for param in ea_template.parameters:
            if random.random() < 0.1:  # Mutation probability per parameter
                if param.param_type == ParameterType.INTEGER:
                    # Gaussian mutation with clamping
                    current_value = mutated_chromosome[param.name]
                    mutation_range = (param.max_value - param.min_value) * 0.1
                    mutated_value = current_value + random.gauss(0, mutation_range)
                    mutated_chromosome[param.name] = int(np.clip(mutated_value, 
                                                               param.min_value, 
                                                               param.max_value))
                elif param.param_type == ParameterType.FLOAT:
                    current_value = mutated_chromosome[param.name]
                    mutation_range = (param.max_value - param.min_value) * 0.1
                    mutated_value = current_value + random.gauss(0, mutation_range)
                    mutated_chromosome[param.name] = np.clip(mutated_value, 
                                                           param.min_value, 
                                                           param.max_value)
                elif param.param_type == ParameterType.BOOLEAN:
                    mutated_chromosome[param.name] = not mutated_chromosome[param.name]
                elif param.param_type == ParameterType.CHOICE:
                    mutated_chromosome[param.name] = random.choice(param.choices)
        
        return Individual(chromosome=mutated_chromosome)

class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimizer"""
    
    def optimize(self, ea_template: EATemplate, 
                objective_function: Callable) -> OptimizationResult:
        """Run grid search optimization"""
        self.start_time = time.time()
        logger.info(f"Starting grid search optimization with resolution {self.config.grid_resolution}")
        
        # Generate grid points
        grid_points = self._generate_grid_points(ea_template)
        total_points = len(grid_points)
        
        logger.info(f"Generated {total_points} grid points to evaluate")
        
        best_fitness = float('-inf')
        best_params = None
        best_objectives = {}
        
        # Evaluate all grid points
        for i, params in enumerate(grid_points):
            if self._check_time_limit():
                break
            
            individual = Individual(chromosome=params)
            individual = self._evaluate_individual(individual, objective_function)
            
            if individual.fitness > best_fitness:
                best_fitness = individual.fitness
                best_params = params.copy()
                best_objectives = individual.objectives.copy()
            
            if i % 100 == 0:
                logger.info(f"Evaluated {i}/{total_points} points, best fitness: {best_fitness:.6f}")
        
        optimization_time = time.time() - self.start_time
        
        result = OptimizationResult(
            ea_id=ea_template.id,
            method=OptimizationMethod.GRID_SEARCH,
            parameters=best_params,
            performance_metrics=best_objectives,
            optimization_time=optimization_time,
            iterations=len(grid_points),
            convergence_achieved=True,
            timestamp=datetime.now()
        )
        
        logger.info(f"Grid search completed: evaluated {len(grid_points)} points, "
                   f"best fitness = {best_fitness:.6f}")
        
        return result
    
    def _generate_grid_points(self, ea_template: EATemplate) -> List[Dict[str, Any]]:
        """Generate all grid points"""
        param_grids = {}
        
        for param in ea_template.parameters:
            if param.param_type == ParameterType.INTEGER:
                param_grids[param.name] = np.linspace(
                    param.min_value, param.max_value, 
                    min(self.config.grid_resolution, param.max_value - param.min_value + 1)
                ).astype(int).tolist()
            elif param.param_type == ParameterType.FLOAT:
                param_grids[param.name] = np.linspace(
                    param.min_value, param.max_value, 
                    self.config.grid_resolution
                ).tolist()
            elif param.param_type == ParameterType.BOOLEAN:
                param_grids[param.name] = [True, False]
            elif param.param_type == ParameterType.CHOICE:
                param_grids[param.name] = param.choices
            else:
                param_grids[param.name] = [param.default_value]
        
        # Generate Cartesian product
        grid_points = []
        self._recursive_grid_generation(param_grids, {}, list(param_grids.keys()), 0, grid_points)
        
        return grid_points
    
    def _recursive_grid_generation(self, param_grids: Dict, current_point: Dict, 
                                  param_names: List[str], depth: int, result: List[Dict]):
        """Recursively generate grid points"""
        if depth == len(param_names):
            result.append(current_point.copy())
            return
        
        param_name = param_names[depth]
        for value in param_grids[param_name]:
            current_point[param_name] = value
            self._recursive_grid_generation(param_grids, current_point, param_names, depth + 1, result)

class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Gaussian Process"""
    
    def optimize(self, ea_template: EATemplate, 
                objective_function: Callable) -> OptimizationResult:
        """Run Bayesian optimization"""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("SciPy required for Bayesian optimization")
        
        self.start_time = time.time()
        logger.info(f"Starting Bayesian optimization with {self.config.bayesian_iterations} iterations")
        
        # Initialize with random samples
        n_initial = min(10, self.config.bayesian_iterations // 5)
        X_samples = []
        y_samples = []
        
        for _ in range(n_initial):
            individual = self._create_random_individual(ea_template)
            individual = self._evaluate_individual(individual, objective_function)
            
            # Convert parameters to numerical array
            x = self._params_to_array(individual.chromosome, ea_template)
            X_samples.append(x)
            y_samples.append(individual.fitness)
        
        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)
        
        best_idx = np.argmax(y_samples)
        best_fitness = y_samples[best_idx]
        best_x = X_samples[best_idx]
        
        # Gaussian Process model
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        # Optimization loop
        for iteration in range(n_initial, self.config.bayesian_iterations):
            if self._check_time_limit():
                break
            
            # Fit GP model
            gp.fit(X_samples, y_samples)
            
            # Acquisition function optimization
            next_x = self._optimize_acquisition(gp, ea_template, best_fitness)
            
            # Evaluate new point
            next_params = self._array_to_params(next_x, ea_template)
            individual = Individual(chromosome=next_params)
            individual = self._evaluate_individual(individual, objective_function)
            
            # Update samples
            X_samples = np.vstack([X_samples, next_x])
            y_samples = np.append(y_samples, individual.fitness)
            
            # Update best
            if individual.fitness > best_fitness:
                best_fitness = individual.fitness
                best_x = next_x
            
            if iteration % 5 == 0:
                logger.info(f"Bayesian iteration {iteration}: Best fitness = {best_fitness:.6f}")
        
        # Convert best solution back to parameters
        best_params = self._array_to_params(best_x, ea_template)
        best_individual = Individual(chromosome=best_params)
        best_individual = self._evaluate_individual(best_individual, objective_function)
        
        optimization_time = time.time() - self.start_time
        
        result = OptimizationResult(
            ea_id=ea_template.id,
            method=OptimizationMethod.BAYESIAN,
            parameters=best_params,
            performance_metrics=best_individual.objectives,
            optimization_time=optimization_time,
            iterations=len(y_samples),
            convergence_achieved=True,
            timestamp=datetime.now()
        )
        
        logger.info(f"Bayesian optimization completed: {len(y_samples)} evaluations, "
                   f"best fitness = {best_fitness:.6f}")
        
        return result
    
    def _params_to_array(self, params: Dict[str, Any], ea_template: EATemplate) -> np.ndarray:
        """Convert parameter dictionary to numerical array"""
        array = []
        for param in ea_template.parameters:
            value = params[param.name]
            
            if param.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                # Normalize to [0, 1]
                normalized = (value - param.min_value) / (param.max_value - param.min_value)
                array.append(normalized)
            elif param.param_type == ParameterType.BOOLEAN:
                array.append(1.0 if value else 0.0)
            elif param.param_type == ParameterType.CHOICE:
                # One-hot encoding
                idx = param.choices.index(value)
                for i in range(len(param.choices)):
                    array.append(1.0 if i == idx else 0.0)
        
        return np.array(array)
    
    def _array_to_params(self, array: np.ndarray, ea_template: EATemplate) -> Dict[str, Any]:
        """Convert numerical array back to parameter dictionary"""
        params = {}
        idx = 0
        
        for param in ea_template.parameters:
            if param.param_type == ParameterType.INTEGER:
                normalized = array[idx]
                value = int(param.min_value + normalized * (param.max_value - param.min_value))
                params[param.name] = np.clip(value, param.min_value, param.max_value)
                idx += 1
            elif param.param_type == ParameterType.FLOAT:
                normalized = array[idx]
                value = param.min_value + normalized * (param.max_value - param.min_value)
                params[param.name] = np.clip(value, param.min_value, param.max_value)
                idx += 1
            elif param.param_type == ParameterType.BOOLEAN:
                params[param.name] = array[idx] > 0.5
                idx += 1
            elif param.param_type == ParameterType.CHOICE:
                choice_probs = array[idx:idx+len(param.choices)]
                choice_idx = np.argmax(choice_probs)
                params[param.name] = param.choices[choice_idx]
                idx += len(param.choices)
        
        return params
    
    def _optimize_acquisition(self, gp, ea_template: EATemplate, best_y: float) -> np.ndarray:
        """Optimize acquisition function (Expected Improvement)"""
        def acquisition(x):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            
            # Expected Improvement
            with np.errstate(divide='warn'):
                improvement = mu - best_y
                Z = improvement / sigma
                ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            
            return -ei[0]  # Minimize negative EI
        
        # Random restart optimization
        best_x = None
        best_val = float('inf')
        
        # Determine parameter space dimensionality
        n_dims = sum([
            1 if param.param_type in [ParameterType.INTEGER, ParameterType.FLOAT, ParameterType.BOOLEAN]
            else len(param.choices) for param in ea_template.parameters
        ])
        
        for _ in range(10):  # 10 random restarts
            x0 = np.random.random(n_dims)
            
            try:
                res = minimize(acquisition, x0, bounds=[(0, 1)] * n_dims, method='L-BFGS-B')
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except:
                continue
        
        if best_x is None:
            best_x = np.random.random(n_dims)
        
        return best_x

class OptimizationEngine:
    """Main optimization engine coordinating different algorithms"""
    
    def __init__(self):
        self.optimizers = {
            OptimizationMethod.GENETIC_ALGORITHM: GeneticOptimizer,
            OptimizationMethod.GRID_SEARCH: GridSearchOptimizer,
            OptimizationMethod.BAYESIAN: BayesianOptimizer,
        }
        
        self.optimization_history = []
        
    def optimize_ea(self, ea_template: EATemplate, 
                   objective_function: Callable,
                   method: OptimizationMethod = OptimizationMethod.GENETIC_ALGORITHM,
                   config: OptimizationConfig = None) -> OptimizationResult:
        """Optimize EA parameters using specified method"""
        
        if config is None:
            config = OptimizationConfig(method=method)
        
        optimizer_class = self.optimizers.get(method)
        if not optimizer_class:
            raise ValueError(f"Unsupported optimization method: {method}")
        
        optimizer = optimizer_class(config)
        
        logger.info(f"Starting optimization of EA {ea_template.name} using {method.value}")
        
        try:
            result = optimizer.optimize(ea_template, objective_function)
            self.optimization_history.append(result)
            
            logger.info(f"Optimization completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    from qnti_ea_generation_engine import EAGenerationEngine, Parameter, ParameterType
    
    # Create sample EA template
    engine = EAGenerationEngine()
    ea = engine.create_ea_template(
        name="Test EA",
        description="Test EA for optimization",
        author="QNTI"
    )
    
    # Add some parameters
    ea.parameters = [
        Parameter("rsi_period", ParameterType.INTEGER, 5, 50, 14),
        Parameter("rsi_overbought", ParameterType.FLOAT, 60.0, 90.0, 70.0),
        Parameter("rsi_oversold", ParameterType.FLOAT, 10.0, 40.0, 30.0),
        Parameter("use_macd", ParameterType.BOOLEAN, default_value=True),
        Parameter("timeframe", ParameterType.CHOICE, choices=["M1", "M5", "M15", "H1"], default_value="H1")
    ]
    
    # Mock objective function
    def mock_objective(params):
        # Simulate backtest metrics based on parameters
        return BacktestMetrics(
            total_return=random.uniform(-0.5, 2.0),
            sharpe_ratio=random.uniform(-1.0, 3.0),
            max_drawdown=random.uniform(-0.5, -0.05),
            win_rate=random.uniform(0.3, 0.8),
            profit_factor=random.uniform(0.5, 3.0)
        )
    
    # Run optimization
    optimization_engine = OptimizationEngine()
    config = OptimizationConfig(
        method=OptimizationMethod.GENETIC_ALGORITHM,
        population_size=20,
        generations=10
    )
    
    result = optimization_engine.optimize_ea(ea, mock_objective, config=config)
    
    print(f"Optimization completed:")
    print(f"Best parameters: {result.parameters}")
    print(f"Performance metrics: {result.performance_metrics}")
    print(f"Optimization time: {result.optimization_time:.2f} seconds")