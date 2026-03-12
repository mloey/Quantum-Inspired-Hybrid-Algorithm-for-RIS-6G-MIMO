"""
Core algorithms module for QI-HFPA-DRL framework
"""

from .quantum_inspired import QuantumSystem
from .flamingo_search import FlamingoSearchAlgorithm
from .pangolin_optimization import PangolinOptimizationAlgorithm
from .hybrid_algorithm import QuantumInspiredHybridFramework

__all__ = [
    'QuantumSystem',
    'FlamingoSearchAlgorithm',
    'PangolinOptimizationAlgorithm',
    'QuantumInspiredHybridFramework',
]
