"""
Flamingo Search Algorithm (FSA)
Mimics flamingo flocking and filter-feeding behaviors in nature.
"""

import numpy as np
from typing import Tuple, List, Callable
import logging

logger = logging.getLogger(__name__)


class FlamingoSearchAlgorithm:
    """
    FSA: Bio-inspired algorithm based on flamingo collective behavior.
    Models flocking patterns and cooperative filter-feeding behaviors.
    """
    
    def __init__(self, population_size: int, dimensions: int, 
                 bounds: List[Tuple[float, float]], max_iterations: int = 100):
        """
        Initialize Flamingo Search Algorithm.
        
        Args:
            population_size: Number of flamingos in flock
            dimensions: Problem dimensionality
            bounds: Variable bounds
            max_iterations: Maximum algorithm iterations
        """
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iterations = max_iterations
        
        # Flamingo flock properties
        self.position = self._initialize_population()
        self.velocity = np.random.uniform(-1, 1, (population_size, dimensions))
        
        # Flocking parameters
        self.separation_weight = 0.3
        self.alignment_weight = 0.4
        self.cohesion_weight = 0.3
        self.filter_feeding_efficiency = np.random.uniform(0.5, 1.0, population_size)
        
        logger.info(f"Initialized FSA with {population_size} flamingos in {dimensions} dimensions")
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize flamingo positions within bounds."""
        population = np.zeros((self.population_size, self.dimensions))
        for i in range(self.dimensions):
            low, high = self.bounds[i]
            population[:, i] = np.random.uniform(low, high, self.population_size)
        return population
    
    def separation(self, i: int) -> np.ndarray:
        """
        Separation behavior: maintain distance from neighbors.
        
        Args:
            i: Flamingo index
            
        Returns:
            Separation velocity component
        """
        separation = np.zeros(self.dimensions)
        for j in range(self.population_size):
            if i != j:
                distance = np.linalg.norm(self.position[i] - self.position[j])
                if distance < 0.1:  # Too close threshold
                    diff = self.position[i] - self.position[j]
                    separation += diff / (distance + 1e-8)
        return separation / max(1, self.population_size - 1)
    
    def alignment(self, i: int) -> np.ndarray:
        """
        Alignment behavior: velocity matching with neighbors.
        
        Args:
            i: Flamingo index
            
        Returns:
            Alignment velocity component
        """
        nvx = np.mean(self.velocity)
        neighborhood_velocity = np.zeros(self.dimensions)
        
        # Find neighbors within perception range
        for j in range(self.population_size):
            if i != j:
                distance = np.linalg.norm(self.position[i] - self.position[j])
                if distance < 1.0:  # Perception range
                    neighborhood_velocity += self.velocity[j]
        
        return neighborhood_velocity / max(1, self.population_size - 1)
    
    def cohesion(self, i: int) -> np.ndarray:
        """
        Cohesion behavior: move toward center of mass of neighbors.
        
        Args:
            i: Flamingo index
            
        Returns:
            Cohesion velocity component
        """
        center_mass = np.mean(self.position, axis=0)
        return (center_mass - self.position[i]) * 0.1
    
    def filter_feeding(self, i: int, fitness: np.ndarray) -> np.ndarray:
        """
        Filter-feeding behavior: search strategy based on food (fitness) detection.
        
        Args:
            i: Flamingo index
            fitness: Fitness values of population
            
        Returns:
            Filter-feeding movement
        """
        # High efficiency flamingos search more aggressively
        efficiency = self.filter_feeding_efficiency[i]
        
        # Find nearby high-fitness solutions
        best_neighbors = np.argsort(-fitness)[:max(1, self.population_size//5)]
        
        feeding_direction = np.zeros(self.dimensions)
        for neighbor_idx in best_neighbors:
            if neighbor_idx != i:
                diff = self.position[neighbor_idx] - self.position[i]
                feeding_direction += diff
        
        feeding_direction = feeding_direction / max(1, len(best_neighbors))
        return efficiency * feeding_direction
    
    def update_positions(self, fitness: np.ndarray, global_best: np.ndarray) -> np.ndarray:
        """
        Update flamingo positions using flocking and feeding behaviors.
        
        Args:
            fitness: Current fitness values
            global_best: Global best position
            
        Returns:
            Updated positions
        """
        for i in range(self.population_size):
            # Calculate flocking behaviors
            sep = self.separation(i)
            ali = self.alignment(i)
            coh = self.cohesion(i)
            feed = self.filter_feeding(i, fitness)
            
            # Combine behaviors with weights
            behavior_vector = (self.separation_weight * sep +
                             self.alignment_weight * ali +
                             self.cohesion_weight * coh +
                             0.3 * feed)
            
            # Update velocity and position
            inertia_weight = 0.7
            self.velocity[i] = (inertia_weight * self.velocity[i] + 
                               0.1 * behavior_vector +
                               0.2 * (global_best - self.position[i]))
            
            self.position[i] += self.velocity[i]
            
            # Enforce bounds
            for j in range(self.dimensions):
                low, high = self.bounds[j]
                self.position[i, j] = np.clip(self.position[i, j], low, high)
        
        return self.position
    
    def update_filtering_efficiency(self, fitness: np.ndarray):
        """
        Update filter-feeding efficiency based on individual success.
        
        Args:
            fitness: Current fitness values
        """
        # Normalize fitness
        fitness_normalized = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-8)
        
        # Improve efficiency for high-fitness flamingos (reward success)
        self.filter_feeding_efficiency = (0.8 * self.filter_feeding_efficiency +
                                         0.2 * fitness_normalized)
        self.filter_feeding_efficiency = np.clip(self.filter_feeding_efficiency, 0.3, 1.0)
