"""
Unit tests for QI-HFPA-DRL algorithms
"""

import numpy as np
import unittest
import sys
sys.path.insert(0, '..')

from src.algorithms import (
    QuantumSystem,
    FlamingoSearchAlgorithm,
    PangolinOptimizationAlgorithm,
)
from src.rl_agents import DDQNAgent
from src.optimization import MultiObjectiveOptimizer


class TestQuantumSystem(unittest.TestCase):
    """Test cases for Quantum System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bounds = [(-5, 5), (-5, 5), (-5, 5)]
        self.quantum = QuantumSystem(
            population_size=10,
            dimensions=3,
            bounds=self.bounds
        )
    
    def test_initialization(self):
        """Test quantum system initialization"""
        self.assertEqual(self.quantum.population_size, 10)
        self.assertEqual(self.quantum.dimensions, 3)
        self.assertEqual(len(self.quantum.bounds), 3)
    
    def test_superposition(self):
        """Test superposition principle"""
        population = self.quantum.apply_superposition()
        self.assertEqual(population.shape, (10, 3))
        self.assertTrue(np.all(population >= 0) and np.all(population <= 1))
    
    def test_entanglement(self):
        """Test quantum entanglement"""
        pop = self.quantum.apply_superposition()
        global_best = np.mean(pop, axis=0)
        entangled = self.quantum.apply_entanglement(pop, global_best)
        self.assertEqual(entangled.shape, pop.shape)
    
    def test_interference(self):
        """Test quantum interference"""
        x = np.random.rand(3)
        y = np.random.rand(3)
        result = self.quantum.apply_quantum_interference(x, y)
        self.assertEqual(len(result), 3)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))


class TestFlamingoAlgorithm(unittest.TestCase):
    """Test cases for Flamingo Search Algorithm"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bounds = [(-5, 5), (-5, 5)]
        self.fsa = FlamingoSearchAlgorithm(
            population_size=20,
            dimensions=2,
            bounds=self.bounds,
            max_iterations=50
        )
    
    def test_initialization(self):
        """Test FSA initialization"""
        self.assertEqual(self.fsa.population_size, 20)
        self.assertEqual(self.fsa.dimensions, 2)
        self.assertEqual(self.fsa.position.shape, (20, 2))
    
    def test_separation(self):
        """Test separation behavior"""
        sep = self.fsa.separation(0)
        self.assertEqual(len(sep), 2)
    
    def test_alignment(self):
        """Test alignment behavior"""
        ali = self.fsa.alignment(0)
        self.assertEqual(len(ali), 2)
    
    def test_update_positions(self):
        """Test position update"""
        fitness = np.random.rand(20)
        global_best = np.random.rand(2) * 10 - 5
        
        new_pos = self.fsa.update_positions(fitness, global_best)
        self.assertEqual(new_pos.shape, (20, 2))
        
        # Check bounds
        for i in range(2):
            low, high = self.bounds[i]
            self.assertTrue(np.all(new_pos[:, i] >= low))
            self.assertTrue(np.all(new_pos[:, i] <= high))


class TestPangolinAlgorithm(unittest.TestCase):
    """Test cases for Pangolin Optimization Algorithm"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bounds = [(-5, 5), (-5, 5)]
        self.poa = PangolinOptimizationAlgorithm(
            population_size=20,
            dimensions=2,
            bounds=self.bounds,
            max_iterations=50
        )
    
    def test_initialization(self):
        """Test POA initialization"""
        self.assertEqual(self.poa.population_size, 20)
        self.assertEqual(self.poa.dimensions, 2)
        self.assertEqual(self.poa.position.shape, (20, 2))
    
    def test_defensive_rolling(self):
        """Test defensive rolling behavior"""
        defense = self.poa.defensive_rolling(0, threat_level=0.7)
        self.assertEqual(len(defense), 2)
    
    def test_intelligent_foraging(self):
        """Test intelligent foraging behavior"""
        fitness = np.random.rand(20)
        global_best = np.random.rand(2) * 10 - 5
        
        forage = self.poa.intelligent_foraging(0, fitness, global_best)
        self.assertEqual(len(forage), 2)
    
    def test_update_positions(self):
        """Test position update"""
        fitness = np.random.rand(20)
        global_best = np.random.rand(2) * 10 - 5
        
        new_pos = self.poa.update_positions(fitness, global_best)
        self.assertEqual(new_pos.shape, (20, 2))
        
        # Check bounds
        for i in range(2):
            low, high = self.bounds[i]
            self.assertTrue(np.all(new_pos[:, i] >= low))
            self.assertTrue(np.all(new_pos[:, i] <= high))


class TestDDQNAgent(unittest.TestCase):
    """Test cases for DDQN Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = DDQNAgent(
            state_size=10,
            action_size=5
        )
    
    def test_initialization(self):
        """Test DDQN initialization"""
        self.assertEqual(self.agent.state_size, 10)
        self.assertEqual(self.agent.action_size, 5)
        self.assertEqual(self.agent.epsilon, 1.0)
    
    def test_action_selection(self):
        """Test action selection"""
        state = np.random.randn(10)
        action = self.agent.select_action(state, training=False)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 5)
    
    def test_experience_storage(self):
        """Test experience storage"""
        state = np.random.randn(10)
        action = 2
        reward = 1.0
        next_state = np.random.randn(10)
        done = False
        
        self.agent.store_experience(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.replay_buffer.buffer), 1)
    
    def test_training(self):
        """Test DDQN training"""
        # Add some experiences
        for _ in range(50):
            state = np.random.randn(10)
            action = np.random.randint(5)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = False
            self.agent.store_experience(state, action, reward, next_state, done)
        
        # Train
        loss = self.agent.train(batch_size=32)
        self.assertGreaterEqual(loss, 0)


class TestMultiObjectiveOptimizer(unittest.TestCase):
    """Test cases for Multi-Objective Optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.obj1 = lambda x: -np.sum(x**2)
        self.obj2 = lambda x: -np.sum((x-1)**2)
        self.bounds = [(-5, 5), (-5, 5)]
        
        self.optimizer = MultiObjectiveOptimizer(
            objectives=[self.obj1, self.obj2],
            bounds=self.bounds,
            population_size=10,
            max_iterations=5,
            num_objectives=2
        )
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertEqual(self.optimizer.num_objectives, 2)
        self.assertEqual(self.optimizer.dimensions, 2)
        self.assertEqual(len(self.optimizer.objectives), 2)
    
    def test_evaluate_population(self):
        """Test population evaluation"""
        population = np.random.rand(10, 2) * 10 - 5
        fitness = self.optimizer.evaluate_population(population)
        
        self.assertEqual(fitness.shape, (10, 2))
        self.assertFalse(np.any(np.isnan(fitness)))


if __name__ == '__main__':
    unittest.main()
