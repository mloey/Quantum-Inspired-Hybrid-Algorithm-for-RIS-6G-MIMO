"""
Example: Multi-objective optimization with custom objectives.
"""

import sys
import numpy as np
sys.path.insert(0, '../src')

from optimization.multi_objective import MultiObjectiveOptimizer


# Define custom objective functions
def sphere(x):
    """Sphere function (minimize)."""
    return -np.sum(x**2)  # Negate for maximization


def rosenbrock(x):
    """Rosenbrock function (minimize)."""
    return -np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def zakharov(x):
    """Zakharov function (minimize)."""
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x)+1) * x)
    return -(sum1 + sum2**2 + sum2**4)


def main():
    """Run multi-objective optimization example."""
    
    print("=" * 70)
    print("QI-HFPA-DRL: Multi-Objective Optimization Example")
    print("=" * 70)
    
    # Problem setup
    dimensions = 10
    bounds = [(-5, 5) for _ in range(dimensions)]
    
    # Initialize optimizer
    optimizer = MultiObjectiveOptimizer(
        objectives=[sphere, rosenbrock, zakharov],
        bounds=bounds,
        population_size=30,
        max_iterations=40,
        num_objectives=3
    )
    
    # Run optimization
    print("\nOptimizing 3 objectives with 10 dimensions...")
    print("-" * 70)
    
    solutions, fitness = optimizer.optimize(verbose=True)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nPareto-optimal solutions: {len(solutions)}")
    
    if len(solutions) > 0:
        # Display top solutions
        print("\nTop 5 Solutions (by first objective):")
        top_idx = np.argsort(-fitness[:, 0])[:min(5, len(solutions))]
        
        for i, idx in enumerate(top_idx):
            print(f"\n  Solution {i+1}:")
            print(f"    Sphere:     {fitness[idx, 0]:.6f}")
            print(f"    Rosenbrock: {fitness[idx, 1]:.6f}")
            print(f"    Zakharov:   {fitness[idx, 2]:.6f}")


if __name__ == "__main__":
    main()
