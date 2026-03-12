"""
Quantum-Inspired Hybrid Framework combining FSA, POA, and RL mechanisms.
"""

import numpy as np
from typing import Tuple, List, Callable, Dict, Optional
import logging

from .quantum_inspired import QuantumSystem
from .flamingo_search import FlamingoSearchAlgorithm
from .pangolin_optimization import PangolinOptimizationAlgorithm

logger = logging.getLogger(__name__)


class QuantumInspiredHybridFramework:
    """
    QI-HFPA-DRL Framework: Hybrid algorithm combining quantum-inspired mechanisms,
    Flamingo Search Algorithm, Pangolin Optimization Algorithm with Deep Q-Networks.
    """
    
    def __init__(self, 
                 population_size: int,
                 dimensions: int,
                 bounds: List[Tuple[float, float]],
                 max_iterations: int = 100,
                 num_objectives: int = 1):
        """
        Initialize QI-HFPA-DRL framework.
        
        Args:
            population_size: Population size
            dimensions: Problem dimensionality
            bounds: Variable bounds
            max_iterations: Maximum iterations
            num_objectives: Number of objectives
        """
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.num_objectives = num_objectives
        
        # Initialize component algorithms
        self.quantum_system = QuantumSystem(population_size, dimensions, bounds)
        self.flamingo_algorithm = FlamingoSearchAlgorithm(population_size, dimensions, bounds)
        self.pangolin_algorithm = PangolinOptimizationAlgorithm(population_size, dimensions, bounds)
        
        # Hybrid control parameters
        self.quantum_weight = 0.4
        self.flamingo_weight = 0.3
        self.pangolin_weight = 0.3
        
        # Archive for non-dominated solutions
        self.pareto_archive = []
        self.pareto_fitness = []
        
        logger.info(f"Initialized QI-HFPA-DRL framework with {population_size} solutions")
    
    def get_hybrid_population(self) -> np.ndarray:
        """
        Get hybrid population using quantum superposition.
        
        Returns:
            Hybrid population
        """
        quantum_pop = self.quantum_system.apply_superposition()
        flamingo_pop = self.flamingo_algorithm.position
        pangolin_pop = self.pangolin_algorithm.position
        
        # Normalize all populations to [0, 1]
        flamingo_norm = (flamingo_pop - np.min(flamingo_pop, axis=0)) / \
                       (np.max(flamingo_pop, axis=0) - np.min(flamingo_pop, axis=0) + 1e-8)
        pangolin_norm = (pangolin_pop - np.min(pangolin_pop, axis=0)) / \
                       (np.max(pangolin_pop, axis=0) - np.min(pangolin_pop, axis=0) + 1e-8)
        
        # Combine populations
        hybrid_pop = (self.quantum_weight * quantum_pop +
                     self.flamingo_weight * flamingo_norm +
                     self.pangolin_weight * pangolin_norm)
        
        # Map from [0, 1] to bounds
        for j in range(self.dimensions):
            low, high = self.bounds[j]
            hybrid_pop[:, j] = low + hybrid_pop[:, j] * (high - low)
        
        return hybrid_pop
    
    def update_weights_adaptive(self, iteration: int):
        """
        Adaptively adjust algorithm weights based on iteration progress.
        
        Args:
            iteration: Current iteration
        """
        progress = iteration / self.max_iterations
        
        # Early phase: emphasize quantum exploration
        # Later phase: emphasize exploitation via flamingo and pangolin
        if progress < 0.3:
            self.quantum_weight = 0.5
            self.flamingo_weight = 0.25
            self.pangolin_weight = 0.25
        elif progress < 0.7:
            self.quantum_weight = 0.4
            self.flamingo_weight = 0.3
            self.pangolin_weight = 0.3
        else:
            self.quantum_weight = 0.3
            self.flamingo_weight = 0.35
            self.pangolin_weight = 0.35
    
    def is_dominated(self, solution: np.ndarray, fitness: np.ndarray,
                    reference_solution: np.ndarray, reference_fitness: np.ndarray) -> bool:
        """
        Check if a solution is dominated by another (Pareto dominance).
        
        Args:
            solution: Current solution
            fitness: Current fitness (to maximize)
            reference_solution: Reference solution
            reference_fitness: Reference fitness
            
        Returns:
            True if dominated
        """
        # For maximization: reference dominates solution if reference >= solution in all objectives
        # and > in at least one
        if np.all(reference_fitness >= fitness) and np.any(reference_fitness > fitness):
            return True
        return False
    
    def update_pareto_archive(self, population: np.ndarray, fitness: np.ndarray):
        """
        Update Pareto-optimal solution archive.
        
        Args:
            population: Current population
            fitness: Current fitness values (shape: [pop_size, num_objectives])
        """
        for i in range(len(population)):
            # Check if solution is dominated by archive members
            dominated = False
            to_remove = []
            
            for j, archived_sol in enumerate(self.pareto_archive):
                if self.is_dominated(population[i], fitness[i], 
                                    archived_sol, self.pareto_fitness[j]):
                    dominated = True
                    break
                elif self.is_dominated(archived_sol, self.pareto_fitness[j],
                                      population[i], fitness[i]):
                    to_remove.append(j)
            
            if not dominated:
                # Add new non-dominated solution
                self.pareto_archive.append(population[i].copy())
                self.pareto_fitness.append(fitness[i].copy())
                
                # Remove dominated solutions
                for idx in sorted(to_remove, reverse=True):
                    self.pareto_archive.pop(idx)
                    self.pareto_fitness.pop(idx)
    
    def calculate_hypervolume(self, reference_point: Optional[np.ndarray] = None) -> float:
        """
        Calculate hypervolume of Pareto front (for validation).
        
        Args:
            reference_point: Reference point for hypervolume calculation
            
        Returns:
            Hypervolume value
        """
        if not self.pareto_fitness:
            return 0.0
        
        if reference_point is None:
            reference_point = np.ones(self.num_objectives) * 2.0
        
        pareto_array = np.array(self.pareto_fitness)
        
        # Simple hypervolume calculation using grid-based approximation
        if self.num_objectives == 1:
            return np.max(pareto_array) if len(pareto_array) > 0 else 0.0
        elif self.num_objectives == 2:
            # 2D hypervolume calculation
            sorted_idx = np.argsort(pareto_array[:, 0])
            sorted_fitness = pareto_array[sorted_idx]
            
            hv = 0.0
            prev_y = 0.0
            for i in range(len(sorted_fitness)):
                hv += (sorted_fitness[i, 0] - (sorted_fitness[i-1, 0] if i > 0 else 0)) * \
                      max(prev_y, sorted_fitness[i, 1])
                prev_y = max(prev_y, sorted_fitness[i, 1])
            
            return hv
        else:
            # Approximate hypervolume for higher dimensions
            return np.prod(np.max(pareto_array, axis=0))
