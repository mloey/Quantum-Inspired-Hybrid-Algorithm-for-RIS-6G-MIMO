"""
Quantum-inspired computing principles for multi-objective optimization.
Implements superposition, entanglement, and quantum interference concepts.
"""

import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class QuantumSystem:
    """
    Quantum-inspired system implementing superposition, entanglement, and interference.
    Uses quantum-encoded population representation for simultaneous exploration.
    """
    
    def __init__(self, population_size: int, dimensions: int, bounds: List[Tuple[float, float]]):
        """
        Initialize quantum system.
        
        Args:
            population_size: Number of quantum states
            dimensions: Problem dimensionality
            bounds: Variable bounds [(min, max), ...]
        """
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = bounds
        
        # Quantum state representation
        self.amplitude = np.random.randn(population_size, dimensions)
        self.phase = np.random.uniform(0, 2*np.pi, (population_size, dimensions))
        self.entanglement_strength = np.ones((population_size, dimensions)) * 0.7
        
        logger.info(f"Initialized quantum system with {population_size} states in {dimensions} dimensions")
    
    def apply_superposition(self) -> np.ndarray:
        """
        Apply superposition principle for simultaneous state exploration.
        
        Returns:
            Population representation (0-1 normalized)
        """
        quantum_states = self.amplitude * np.cos(self.phase)
        population = (quantum_states + 1) / 2  # Normalize to [0, 1]
        return np.clip(population, 0, 1)
    
    def apply_entanglement(self, population: np.ndarray, global_best: np.ndarray) -> np.ndarray:
        """
        Apply quantum entanglement for diversity preservation.
        Creates correlation between solution components.
        
        Args:
            population: Current population
            global_best: Global best solution
            
        Returns:
            Entangled population
        """
        entangled = population.copy()
        for i in range(self.population_size):
            correlation = np.dot(population[i], global_best) / (np.linalg.norm(population[i]) + 1e-8)
            entangled[i] = (1 - self.entanglement_strength[i]) * population[i] + \
                          self.entanglement_strength[i] * global_best * correlation
        return entangled
    
    def apply_quantum_interference(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply quantum interference for constructive/destructive combination.
        
        Args:
            x: First solution vector
            y: Second solution vector
            
        Returns:
            Interfered solution
        """
        # Calculate phase difference
        phase_diff = np.abs(self.phase[0] - self.phase[1])
        
        # Constructive interference amplifies similar components
        # Destructive interference cancels dissimilar components
        interference = x * np.cos(phase_diff) + y * np.sin(phase_diff)
        return np.clip(interference, 0, 1)
    
    def update_quantum_state(self, population: np.ndarray, fitness: np.ndarray, 
                            learning_rate: float = 0.1):
        """
        Update quantum state based on fitness landscape.
        
        Args:
            population: Current population
            fitness: Fitness values
            learning_rate: Update rate
        """
        # Update amplitude based on fitness ranking
        fitness_rank = np.argsort(-fitness)
        for idx, rank in enumerate(fitness_rank):
            improvement = (rank + 1) / self.population_size
            self.amplitude[idx] = self.amplitude[idx] * (1 - learning_rate) + \
                                 improvement * learning_rate
        
        # Update phase based on fitness gradient
        for i in range(self.dimensions):
            if i < len(fitness) - 1:
                phase_gradient = (fitness[i] - fitness[i + 1]) / (np.abs(fitness[i]) + 1e-8)
                self.phase[:, i] += learning_rate * phase_gradient
    
    def measure_collapse(self, population: np.ndarray) -> np.ndarray:
        """
        Collapse quantum superposition to classical solution (measurement).
        
        Args:
            population: Superposition state
            
        Returns:
            Classical measurement result
        """
        # Probabilistic collapse based on amplitude
        collapsed = np.zeros_like(population)
        for i in range(self.population_size):
            for j in range(self.dimensions):
                prob = (self.amplitude[i, j] ** 2) / (np.sum(self.amplitude[i] ** 2) + 1e-8)
                collapsed[i, j] = 1 if np.random.random() < prob else 0
        return collapsed
