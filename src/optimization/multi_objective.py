"""
Multi-objective optimization framework using QI-HFPA-DRL.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Optional
import logging

from ..algorithms import QuantumInspiredHybridFramework
from ..rl_agents import RLParameterController

logger = logging.getLogger(__name__)


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization framework combining QI-HFPA-DRL with RL-based parameter control.
    """
    
    def __init__(self, 
                 objectives: List[Callable],
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 max_iterations: int = 100,
                 num_objectives: int = None):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objectives: List of objective functions to maximize
            bounds: Variable bounds
            population_size: Population size
            max_iterations: Maximum iterations
            num_objectives: Number of objectives (auto-detected if None)
        """
        self.objectives = objectives
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.num_objectives = num_objectives or len(objectives)
        self.dimensions = len(bounds)
        
        # Initialize hybrid framework
        self.framework = QuantumInspiredHybridFramework(
            population_size=population_size,
            dimensions=self.dimensions,
            bounds=bounds,
            max_iterations=max_iterations,
            num_objectives=self.num_objectives
        )
        
        # Initialize RL parameter controller
        param_ranges = {
            'quantum_weight': (0.2, 0.6),
            'flamingo_weight': (0.2, 0.4),
            'pangolin_weight': (0.2, 0.4),
            'inertia': (0.5, 0.9),
            'cognitive': (0.1, 1.5),
        }
        
        self.rl_controller = RLParameterController(
            parameter_names=list(param_ranges.keys()),
            parameter_ranges=param_ranges,
            num_objectives=self.num_objectives
        )
        
        # Optimization history
        self.fitness_history = []
        self.diversity_history = []
        self.best_fitness = np.full(self.num_objectives, -np.inf)
        self.best_solution = None
        
        logger.info(f"Initialized MultiObjectiveOptimizer with {self.num_objectives} objectives")
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate entire population across all objectives.
        
        Args:
            population: Population array
            
        Returns:
            Fitness array [pop_size, num_objectives]
        """
        fitness = np.zeros((len(population), self.num_objectives))
        
        for i in range(len(population)):
            for j, objective in enumerate(self.objectives):
                try:
                    fitness[i, j] = objective(population[i])
                except Exception as e:
                    logger.warning(f"Error evaluating objective {j} for solution {i}: {e}")
                    fitness[i, j] = -np.inf
        
        return fitness
    
    def calculate_diversity(self, population: np.ndarray) -> float:
        """
        Calculate population diversity.
        
        Args:
            population: Current population
            
        Returns:
            Diversity metric
        """
        # Average distance between solutions
        distances = []
        for i in range(min(len(population), 10)):  # Check subset for efficiency
            for j in range(i+1, min(len(population), 10)):
                dist = np.linalg.norm(population[i] - population[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute multi-objective optimization.
        
        Args:
            verbose: Print progress information
            
        Returns:
            Best solutions (Pareto front), Best fitness values
        """
        # Initialize population
        population = self.framework.get_hybrid_population()
        fitness = self.evaluate_population(population)
        
        # Track best solutions
        if self.num_objectives == 1:
            best_idx = np.argmax(fitness[:, 0])
            self.best_fitness = fitness[best_idx]
            self.best_solution = population[best_idx].copy()
        else:
            self.framework.update_pareto_archive(population, fitness)
        
        iteration = 0
        for iteration in range(self.max_iterations):
            # Get RL controller state and action
            state = self.rl_controller.extract_state(
                population, 
                np.max(fitness, axis=1) if self.num_objectives > 1 else fitness,
                iteration, 
                self.max_iterations
            )
            
            action = self.rl_controller.select_action(state, training=True)
            params = self.rl_controller.apply_action(action)
            
            # Update framework weights
            self.framework.quantum_weight = params.get('quantum_weight', 0.4)
            self.framework.flamingo_weight = params.get('flamingo_weight', 0.3)
            self.framework.pangolin_weight = params.get('pangolin_weight', 0.3)
            
            # Update algorithms
            self.framework.flamingo_algorithm.update_positions(
                fitness[:, 0] if self.num_objectives == 1 else np.max(fitness, axis=1),
                self.best_solution if self.best_solution is not None else population[0]
            )
            
            self.framework.pangolin_algorithm.update_positions(
                fitness[:, 0] if self.num_objectives == 1 else np.max(fitness, axis=1),
                self.best_solution if self.best_solution is not None else population[0]
            )
            
            # Update quantum system
            self.framework.quantum_system.update_quantum_state(
                population,
                fitness[:, 0] if self.num_objectives == 1 else np.max(fitness, axis=1),
                learning_rate=0.1
            )
            
            # Get new population
            population = self.framework.get_hybrid_population()
            new_fitness = self.evaluate_population(population)
            
            # Calculate reward
            avg_fitness_before = np.mean(fitness)
            avg_fitness_after = np.mean(new_fitness)
            diversity_before = self.calculate_diversity(population)
            
            fitness = new_fitness
            diversity_after = self.calculate_diversity(population)
            
            reward = self.rl_controller.calculate_reward(
                avg_fitness_before, avg_fitness_after,
                diversity_before, diversity_after
            )
            
            # Update RL agent
            self.framework.quantum_system.update_quantum_state(
                population, fitness[:, 0] if self.num_objectives == 1 else np.max(fitness, axis=1)
            )
            next_state = self.rl_controller.extract_state(
                population, np.max(fitness, axis=1) if self.num_objectives > 1 else fitness,
                iteration + 1, self.max_iterations
            )
            
            self.rl_controller.update(state, action, reward, next_state,
                                     done=(iteration == self.max_iterations - 1))
            
            # Update best solutions
            if self.num_objectives == 1:
                best_idx = np.argmax(fitness[:, 0])
                if fitness[best_idx, 0] > self.best_fitness[0]:
                    self.best_fitness = fitness[best_idx]
                    self.best_solution = population[best_idx].copy()
            else:
                self.framework.update_pareto_archive(population, fitness)
            
            # Store history
            self.fitness_history.append(np.mean(fitness))
            self.diversity_history.append(diversity_after)
            
            if verbose and (iteration + 1) % 10 == 0:
                if self.num_objectives == 1:
                    logger.info(f"Iteration {iteration+1}: Best Fitness = {self.best_fitness[0]:.6f}")
                else:
                    hv = self.framework.calculate_hypervolume()
                    logger.info(f"Iteration {iteration+1}: Pareto Front Size = {len(self.framework.pareto_archive)}, HV = {hv:.6f}")
        
        # Return results
        if self.num_objectives == 1:
            return np.array([self.best_solution]), np.array([self.best_fitness])
        else:
            return np.array(self.framework.pareto_archive), np.array(self.framework.pareto_fitness)
