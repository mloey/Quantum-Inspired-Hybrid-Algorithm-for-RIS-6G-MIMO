"""
Quantum-Inspired Hybrid Flamingo-Pangolin Algorithm with Deep Reinforcement Learning
for RIS-assisted 6G MIMO Optimization
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__description__ = "QI-HFPA-DRL: Advanced multi-objective optimization for 6G systems"

from .algorithms import (
    FlamingoSearchAlgorithm,
    PangolinOptimizationAlgorithm,
    QuantumInspiredHybridFramework
)
from .rl_agents import DDQNAgent, RLParameterController
from .optimization import MultiObjectiveOptimizer, RISAntennaNAOptimizer

__all__ = [
    'FlamingoSearchAlgorithm',
    'PangolinOptimizationAlgorithm',
    'QuantumInspiredHybridFramework',
    'DDQNAgent',
    'RLParameterController',
    'MultiObjectiveOptimizer',
    'RISAntennaNAOptimizer',
]
