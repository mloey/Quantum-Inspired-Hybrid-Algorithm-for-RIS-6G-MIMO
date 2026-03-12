# QI-HFPA-DRL: Quantum-Inspired Hybrid Flamingo-Pangolin Algorithm with Deep Reinforcement Learning

## Overview

**QI-HFPA-DRL** is a novel hybrid metaheuristic framework for comprehensive multi-objective optimization of RIS-assisted 6G MIMO antenna systems. The algorithm synergistically integrates:

- **Quantum Computing Principles**: Superposition, entanglement, and quantum interference for enhanced exploration
- **Flamingo Search Algorithm (FSA)**: Bio-inspired flocking and filter-feeding behaviors
- **Pangolin Optimization Algorithm (POA)**: Defensive rolling and intelligent foraging strategies
- **Deep Reinforcement Learning**: Double Deep Q-Networks (DDQN) for adaptive parameter control

## Key Features

### 🎯 Core Algorithms
- **Quantum-Inspired Population Representation**: Simultaneous exploration of multiple antenna configuration states
- **Flamingo Flocking Behaviors**: Separation, alignment, cohesion, and filter-feeding mechanisms
- **Pangolin Protective Strategies**: Defensive rolling and pheromone-based foraging
- **Adaptive Parameter Control**: DDQN-based dynamic strategy selection

### 📊 Multi-Objective Optimization
- Spectral Efficiency Maximization
- Energy Efficiency Optimization
- Beam Steering Accuracy
- Sidelobe Level Suppression
- Coverage Probability Enhancement
- Interference Mitigation
- Hardware Complexity Reduction

### 🚀 Performance Metrics
- improvement in hypervolume indicator
- reduction in inverted generational distance
- better spacing uniformity
- faster convergence rate

### 🔬 Validation
- Wireless applications: beamforming, channel estimation, RIS optimization, UAV trajectory, resource allocation

## Project Structure

```
qi_hfpa_drl/
├── src/
│   ├── algorithms/
│   │   ├── quantum_inspired.py       # Quantum system implementation
│   │   ├── flamingo_search.py        # FSA implementation
│   │   ├── pangolin_optimization.py  # POA implementation
│   │   └── hybrid_algorithm.py       # Hybrid framework
│   ├── rl_agents/
│   │   ├── ddqn.py                   # Double Deep Q-Network agent
│   │   └── rl_controller.py          # RL parameter controller
│   ├── optimization/
│   │   ├── multi_objective.py        # Multi-objective optimizer
│   │   └── ris_antenna.py            # RIS-MIMO antenna optimizer
│   └── utils/
│       ├── metrics.py                # Performance metrics
│       └── visualization.py          # Plotting utilities
├── examples/
│   ├── ris_mimo_optimization.py     # RIS-MIMO example
│   └── multi_objective_example.py   # Basic multi-objective example
├── tests/
│   └── test_algorithms.py           # Unit 
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package setup
└── README.md                       # This file
```

## Installation

### Prerequisites
- Python 3.8+
- NumPy
- (Optional) Matplotlib for visualization

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mloey/Quantum-Inspired-Hybrid-Algorithm-for-RIS-6G-MIMO
cd Quantum-Inspired-Hybrid-Algorithm-for-RIS-6G-MIMO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

