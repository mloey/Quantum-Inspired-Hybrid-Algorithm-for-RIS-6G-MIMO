"""
Example: RIS-assisted MIMO antenna optimization using QI-HFPA-DRL.
"""

import sys
import numpy as np
sys.path.insert(0, '../src')

from optimization.ris_antenna import RISAntennaNAOptimizer


def main():
    """Run RIS-MIMO antenna optimization example."""
    
    print("=" * 70)
    print("QI-HFPA-DRL: RIS-Assisted 6G MIMO Antenna Optimization")
    print("=" * 70)
    
    # Initialize optimizer
    optimizer = RISAntennaNAOptimizer(
        num_ris_elements=256,
        num_antenna_elements=64,
        frequency=28.0,  # 28 GHz
        population_size=50,
        max_iterations=50  # More iterations for production
    )
    
    # Run optimization
    print("\nStarting multi-objective optimization...")
    print("-" * 70)
    
    solutions, fitness = optimizer.optimize(verbose=True)
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    if len(solutions) > 0:
        print(f"\nPareto-optimal solutions found: {len(solutions)}")
        
        # Display best solution
        best_idx = np.argmax(fitness[:, 0])
        best_solution = solutions[best_idx]
        best_fitness = fitness[best_idx]
        
        print(f"\nBest Solution Metrics:")
        print(f"  Spectral Efficiency:    {best_fitness[0]:.2f} bps/Hz")
        print(f"  Energy Efficiency:      {best_fitness[1]:.2f} Mbits/Joule")
        print(f"  Beam Steering Accuracy: {best_fitness[2]:.4f}")
        print(f"  Sidelobe Suppression:   {best_fitness[3]:.4f} dB")
        print(f"  Coverage Probability:   {best_fitness[4]:.4f}")
        
        # Decode best solution
        params = optimizer.decode_solution(best_solution)
        
        print(f"\nBest Configuration:")
        print(f"  RIS Phase Range: [{np.min(params['ris_phases']):.4f}, {np.max(params['ris_phases']):.4f}]")
        print(f"  Antenna Weight Range: [{np.min(params['antenna_weights']):.4f}, {np.max(params['antenna_weights']):.4f}]")
        print(f"  Antenna Angle Range: [{np.min(params['antenna_angles']):.4f}, {np.max(params['antenna_angles']):.4f}] rad")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
