"""
Pangolin Optimization Algorithm (POA)
Inspired by pangolin defensive rolling and foraging strategies.
"""

import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class PangolinOptimizationAlgorithm:
    """
    POA: Bio-inspired algorithm based on pangolin protective and foraging behaviors.
    Models defensive rolling mechanics and intelligent foraging strategies.
    """
    
    def __init__(self, population_size: int, dimensions: int,
                 bounds: List[Tuple[float, float]], max_iterations: int = 100):
        """
        Initialize Pangolin Optimization Algorithm.
        
        Args:
            population_size: Number of pangolins
            dimensions: Problem dimensionality
            bounds: Variable bounds
            max_iterations: Maximum iterations
        """
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iterations = max_iterations
        
        # Pangolin properties
        self.position = self._initialize_population()
        self.velocity = np.random.uniform(-1, 1, (population_size, dimensions))
        self.roll_state = np.zeros(population_size)  # Defensive state
        self.foraging_memory = np.copy(self.position)
        
        # Behavioral parameters
        self.defense_threshold = 0.5
        self.foraging_efficiency = np.random.uniform(0.6, 1.0, population_size)
        
        logger.info(f"Initialized POA with {population_size} pangolins in {dimensions} dimensions")
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize pangolin positions within bounds."""
        population = np.zeros((self.population_size, self.dimensions))
        for i in range(self.dimensions):
            low, high = self.bounds[i]
            population[:, i] = np.random.uniform(low, high, self.population_size)
        return population
    
    def defensive_rolling(self, i: int, threat_level: float) -> np.ndarray:
        """
        Defensive rolling behavior: curl into a protective ball when threatened.
        Reduces exploration, increases exploitation in promising regions.
        
        Args:
            i: Pangolin index
            threat_level: How threatened the solution space is (0-1)
            
        Returns:
            Adjusted position from defensive rolling
        """
        if threat_level > self.defense_threshold:
            # Enter defensive mode: reduce movement
            self.roll_state[i] = min(1.0, self.roll_state[i] + 0.1)
            
            # Contract toward best known position (foraging memory)
            contraction_force = (self.foraging_memory[i] - self.position[i]) * self.roll_state[i]
            return contraction_force
        else:
            # Exit defensive mode gradually
            self.roll_state[i] = max(0.0, self.roll_state[i] - 0.05)
            return np.zeros(self.dimensions)
    
    def intelligent_foraging(self, i: int, fitness: np.ndarray, global_best: np.ndarray) -> np.ndarray:
        """
        Intelligent foraging behavior: search for food sources (optimal solutions).
        Pangolins use olfactory senses to follow pheromone trails.
        
        Args:
            i: Pangolin index
            fitness: Current fitness values
            global_best: Global best solution
            
        Returns:
            Foraging direction
        """
        efficiency = self.foraging_efficiency[i]
        
        # Pheromone trail: stronger attraction to better solutions
        pheromone_strength = np.zeros(self.dimensions)
        
        # Find elite solutions (top 20%)
        elite_num = max(1, self.population_size // 5)
        elite_indices = np.argsort(-fitness)[:elite_num]
        
        # Follow pheromone trails from elite solutions
        for idx in elite_indices:
            if idx != i:
                trail = self.position[idx] - self.position[i]
                pheromone_strength += trail * (fitness[idx] / (np.max(fitness) + 1e-8))
        
        pheromone_strength = pheromone_strength / max(1, elite_num)
        
        # Scale by individual efficiency
        return efficiency * pheromone_strength * 0.1
    
    def tongue_flicking(self, i: int, fitness: np.ndarray) -> np.ndarray:
        """
        Tongue-flicking behavior: sense local environment for food sources.
        Small local explorations to detect promising nearby regions.
        
        Args:
            i: Pangolin index
            fitness: Fitness values
            
        Returns:
            Local search direction
        """
        # Generate local search directions (like tongue flicking)
        local_search = np.random.randn(self.dimensions) * 0.05
        
        # Bias toward regions with better fitness
        nearest_neighbors_count = max(1, self.population_size // 3)
        distances = np.linalg.norm(self.position - self.position[i], axis=1)
        nearest_indices = np.argsort(distances)[1:nearest_neighbors_count+1]
        
        best_neighbor = nearest_indices[np.argmax(fitness[nearest_indices])]
        direction = self.position[best_neighbor] - self.position[i]
        
        if np.linalg.norm(direction) > 1e-8:
            direction = direction / np.linalg.norm(direction)
        
        return local_search + 0.15 * direction
    
    def update_positions(self, fitness: np.ndarray, global_best: np.ndarray) -> np.ndarray:
        """
        Update pangolin positions using defensive and foraging behaviors.
        
        Args:
            fitness: Current fitness values
            global_best: Global best position
            
        Returns:
            Updated positions
        """
        # Calculate threat level (environmental pressure)
        fitness_diversity = np.std(fitness) / (np.mean(fitness) + 1e-8)
        threat_level = 1.0 - fitness_diversity  # Low diversity = high threat
        threat_level = np.clip(threat_level, 0, 1)
        
        for i in range(self.population_size):
            # Calculate behaviors
            defense = self.defensive_rolling(i, threat_level)
            forage = self.intelligent_foraging(i, fitness, global_best)
            tongue = self.tongue_flicking(i, fitness)
            
            # Combine behaviors
            inertia_weight = 0.7 * (1 - self.roll_state[i])  # Reduced inertia when rolling
            
            self.velocity[i] = (inertia_weight * self.velocity[i] +
                               forage + tongue + 0.1 * defense +
                               0.15 * (global_best - self.position[i]))
            
            self.position[i] += self.velocity[i]
            
            # Enforce bounds
            for j in range(self.dimensions):
                low, high = self.bounds[j]
                self.position[i, j] = np.clip(self.position[i, j], low, high)
            
            # Update foraging memory (best position found)
            if fitness[i] > np.max(fitness[np.nonzero(self.foraging_memory != self.position)]):
                self.foraging_memory[i] = self.position[i].copy()
        
        return self.position
    
    def update_foraging_efficiency(self, fitness: np.ndarray):
        """
        Update foraging efficiency based on success in finding food sources.
        
        Args:
            fitness: Current fitness values
        """
        # Normalize fitness
        if np.max(fitness) - np.min(fitness) > 1e-8:
            fitness_normalized = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness))
        else:
            fitness_normalized = np.ones_like(fitness)
        
        # Reward successful foragers
        self.foraging_efficiency = (0.8 * self.foraging_efficiency +
                                   0.2 * fitness_normalized)
        self.foraging_efficiency = np.clip(self.foraging_efficiency, 0.4, 1.0)
