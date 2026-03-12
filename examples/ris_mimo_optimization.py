"""
Example: Basic RIS-MIMO antenna optimization using QI-HFPA-DRL
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from src.optimization import RISAntennaNAOptimizer


def main():
    """Run RIS-MIMO antenna optimization example."""
    
    print("=" * 80)
    print("QI-HFPA-DRL: RIS-assisted 6G MIMO Antenna Optimization")
    print("=" * 80)
    print()
    
    # Initialize optimizer
    print("Initializing RIS-MIMO Optimizer...")
    optimizer = RISAntennaNAOptimizer(
        num_ris_elements=256,
        num_antenna_elements=64,
        frequency=28.0,  # 28 GHz millimeter-wave band
        population_size=30,
        max_iterations=50  # Reduced for example
    )
    print(f"Optimizer initialized with {256} RIS elements and {64} antenna elements")
    print()
    
    # Run optimization
    print("Starting multi-objective optimization...")
    print("Optimizing: Spectral Efficiency, Energy Efficiency, Beam Accuracy,")
    print("            Sidelobe Suppression, Coverage Probability")
    print()
    
    solutions, fitness = optimizer.optimize(verbose=True)
    
    print()
    print("=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    if len(solutions) > 0:
        print(f"\nFound {len(solutions)} Pareto-optimal solutions")
        print("\nTop 3 Solutions:")
        print("-" * 80)
        
        for i in range(min(3, len(solutions))):
            print(f"\nSolution {i+1}:")
            print(f"  Spectral Efficiency:    {fitness[i, 0]:8.2f} bps/Hz")
            print(f"  Energy Efficiency:      {fitness[i, 1]:8.2f} Mbits/Joule")
            print(f"  Beam Steering Accuracy: {fitness[i, 2]:8.4f}")
            print(f"  Sidelobe Suppression:   {fitness[i, 3]:8.2f} dB")
            print(f"  Coverage Probability:   {fitness[i, 4]:8.4f}")
    
    print("\n" + "=" * 80)
    print("Optimization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
