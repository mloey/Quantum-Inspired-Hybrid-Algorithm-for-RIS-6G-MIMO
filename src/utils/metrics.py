"""
Performance metrics for multi-objective optimization algorithms.
"""

import numpy as np
from typing import List, Tuple


def calculate_hypervolume(solutions: np.ndarray, reference_point: np.ndarray = None) -> float:
    """
    Calculate hypervolume indicator for Pareto front.
    
    Args:
        solutions: Pareto front solutions [n_solutions, n_objectives]
        reference_point: Reference point for HV calculation
        
    Returns:
        Hypervolume value
    """
    if solutions.size == 0:
        return 0.0
    
    n_objectives = solutions.shape[1] if solutions.ndim > 1 else 1
    
    if reference_point is None:
        reference_point = np.ones(n_objectives) * 2.0
    
    if n_objectives == 1:
        return np.max(solutions) if solutions.size > 0 else 0.0
    elif n_objectives == 2:
        solutions = np.atleast_2d(solutions)
        sorted_idx = np.argsort(solutions[:, 0])
        sorted_solutions = solutions[sorted_idx]
        
        hv = 0.0
        prev_y = 0.0
        for i in range(len(sorted_solutions)):
            x_width = sorted_solutions[i, 0] - (sorted_solutions[i-1, 0] if i > 0 else 0)
            y_height = max(prev_y, sorted_solutions[i, 1])
            hv += x_width * y_height
            prev_y = max(prev_y, sorted_solutions[i, 1])
        
        return hv
    else:
        # Approximate for higher dimensions
        return np.prod(np.max(solutions, axis=0))


def inverted_generational_distance(pareto_front: np.ndarray, 
                                  reference_front: np.ndarray) -> float:
    """
    Calculate Inverted Generational Distance (IGD).
    
    Args:
        pareto_front: Obtained Pareto front
        reference_front: Reference Pareto front
        
    Returns:
        IGD value (lower is better)
    """
    if pareto_front.size == 0:
        return np.inf
    
    pareto_front = np.atleast_2d(pareto_front)
    reference_front = np.atleast_2d(reference_front)
    
    igd = 0.0
    for ref_point in reference_front:
        distances = np.linalg.norm(pareto_front - ref_point, axis=1)
        igd += np.min(distances)
    
    return igd / len(reference_front)


def spacing_uniformity(pareto_front: np.ndarray) -> float:
    """
    Calculate spacing uniformity of Pareto front solutions.
    
    Args:
        pareto_front: Pareto front solutions
        
    Returns:
        Spacing metric (higher is better for uniformity)
    """
    if len(pareto_front) < 2:
        return 1.0
    
    pareto_front = np.atleast_2d(pareto_front)
    
    # Calculate distances between consecutive solutions
    distances = []
    for i in range(len(pareto_front) - 1):
        dist = np.linalg.norm(pareto_front[i+1] - pareto_front[i])
        distances.append(dist)
    
    if not distances:
        return 1.0
    
    distances = np.array(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Uniformity: coefficient of variation (lower = more uniform)
    uniformity = 1.0 / (1.0 + std_dist / (mean_dist + 1e-8))
    
    return uniformity


def convergence_rate(fitness_history: List[float], window_size: int = 10) -> float:
    """
    Calculate convergence rate of optimization algorithm.
    
    Args:
        fitness_history: Historical fitness values
        window_size: Window for rate calculation
        
    Returns:
        Convergence rate (improvements per iteration)
    """
    if len(fitness_history) < window_size:
        return 0.0
    
    recent = np.array(fitness_history[-window_size:])
    early = np.array(fitness_history[:window_size])
    
    improvement = np.mean(recent) - np.mean(early)
    rate = improvement / window_size
    
    return rate
