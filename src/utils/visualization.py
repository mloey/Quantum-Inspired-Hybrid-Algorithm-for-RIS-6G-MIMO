"""
Visualization utilities for optimization results.
"""

import numpy as np
from typing import List, Optional

# Mock matplotlib since it might not be installed
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_pareto_front(solutions: np.ndarray, objectives_idx: List[int] = None,
                      title: str = "Pareto Front", save_path: Optional[str] = None):
    """
    Plot Pareto front in 2D or 3D.
    
    Args:
        solutions: Solutions array [n_solutions, n_objectives]
        objectives_idx: Indices of objectives to plot
        title: Plot title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    solutions = np.atleast_2d(solutions)
    
    if objectives_idx is None:
        objectives_idx = [0, 1] if solutions.shape[1] >= 2 else [0]
    
    if len(objectives_idx) == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(solutions[:, objectives_idx[0]], solutions[:, objectives_idx[1]], 
                   s=50, alpha=0.6, edgecolors='black')
        plt.xlabel(f'Objective {objectives_idx[0]}')
        plt.ylabel(f'Objective {objectives_idx[1]}')
        plt.title(title)
        plt.grid(True, alpha=0.3)
    else:
        print(f"Cannot plot {len(objectives_idx)} objectives in 2D")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_convergence(fitness_history: List[float], title: str = "Convergence",
                    save_path: Optional[str] = None):
    """
    Plot algorithm convergence curve.
    
    Args:
        fitness_history: Historical fitness values
        title: Plot title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_diversity(diversity_history: List[float], title: str = "Population Diversity",
                  save_path: Optional[str] = None):
    """
    Plot population diversity over iterations.
    
    Args:
        diversity_history: Historical diversity values
        title: Plot title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(diversity_history, linewidth=2, color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Diversity')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
