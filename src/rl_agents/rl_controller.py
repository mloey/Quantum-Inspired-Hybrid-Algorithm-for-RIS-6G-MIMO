"""
Reinforcement Learning Parameter Controller for adaptive algorithm tuning.
"""

import numpy as np
from typing import Dict, Tuple
import logging

from .ddqn import DDQNAgent

logger = logging.getLogger(__name__)


class RLParameterController:
    """
    RL-based controller for automatic hyperparameter adaptation in optimization.
    Uses DDQN to adjust algorithm parameters based on optimization landscape.
    """
    
    def __init__(self, parameter_names: list, parameter_ranges: Dict[str, Tuple[float, float]],
                 num_objectives: int = 1):
        """
        Initialize RL parameter controller.
        
        Args:
            parameter_names: Names of parameters to control
            parameter_ranges: Dictionary of parameter bounds
            num_objectives: Number of optimization objectives
        """
        self.parameter_names = parameter_names
        self.parameter_ranges = parameter_ranges
        self.num_objectives = num_objectives
        
        # State features: convergence metrics, diversity, fitness statistics
        self.state_size = 10  # Configurable
        self.action_size = len(parameter_names) * 3  # 3 actions per parameter (increase, decrease, maintain)
        
        # DDQN agent
        self.agent = DDQNAgent(self.state_size, self.action_size)
        
        # Current parameters
        self.current_parameters = {}
        for param, (low, high) in parameter_ranges.items():
            self.current_parameters[param] = (low + high) / 2
        
        # History for state calculation
        self.fitness_history = []
        self.diversity_history = []
        
        logger.info(f"Initialized RL Parameter Controller with {len(parameter_names)} parameters")
    
    def extract_state(self, population: np.ndarray, fitness: np.ndarray, 
                     iteration: int, max_iterations: int) -> np.ndarray:
        """
        Extract state features from current optimization state.
        
        Args:
            population: Current population
            fitness: Current fitness values
            iteration: Current iteration
            max_iterations: Maximum iterations
            
        Returns:
            State vector
        """
        state = np.zeros(self.state_size)
        
        # Feature 1-2: Fitness statistics
        state[0] = np.mean(fitness)
        state[1] = np.std(fitness) if np.std(fitness) > 0 else 1.0
        
        # Feature 3-4: Population diversity
        pop_diversity = np.mean([np.std(population[:, i]) for i in range(population.shape[1])])
        state[2] = pop_diversity
        state[3] = np.std(fitness) / (np.mean(np.abs(fitness)) + 1e-8)  # Convergence rate
        
        # Feature 5: Progress indicator
        progress = iteration / max_iterations
        state[4] = progress
        
        # Feature 6-8: Fitness trend
        self.fitness_history.append(np.mean(fitness))
        if len(self.fitness_history) > 1:
            state[5] = (self.fitness_history[-1] - self.fitness_history[-2]) / \
                      (np.abs(self.fitness_history[-2]) + 1e-8)
        
        if len(self.fitness_history) > 3:
            recent_improvement = np.mean([self.fitness_history[-i] - self.fitness_history[-i-1] 
                                         for i in range(1, min(4, len(self.fitness_history)))])
            state[6] = recent_improvement
        
        # Feature 9-10: Diversity metrics
        self.diversity_history.append(pop_diversity)
        if len(self.diversity_history) > 1:
            state[7] = (self.diversity_history[-1] - self.diversity_history[-2]) / \
                      (self.diversity_history[-2] + 1e-8)
        
        # Normalize state
        state = np.clip(state, -10, 10)
        
        return state
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select parameter adjustment action.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action index
        """
        return self.agent.select_action(state, training=training)
    
    def apply_action(self, action: int) -> Dict[str, float]:
        """
        Apply action to adjust parameters.
        
        Args:
            action: Action index
            
        Returns:
            Updated parameters dictionary
        """
        param_idx = action // 3
        action_type = action % 3
        
        if param_idx >= len(self.parameter_names):
            return self.current_parameters
        
        param_name = self.parameter_names[param_idx]
        low, high = self.parameter_ranges[param_name]
        current = self.current_parameters[param_name]
        
        # Adjust parameter
        adjustment_step = (high - low) * 0.1
        
        if action_type == 0:  # Increase
            self.current_parameters[param_name] = min(high, current + adjustment_step)
        elif action_type == 1:  # Decrease
            self.current_parameters[param_name] = max(low, current - adjustment_step)
        else:  # Maintain
            pass
        
        return self.current_parameters
    
    def calculate_reward(self, fitness_before: float, fitness_after: float,
                        diversity_before: float, diversity_after: float) -> float:
        """
        Calculate reward for RL training.
        
        Args:
            fitness_before: Fitness before action
            fitness_after: Fitness after action
            diversity_before: Diversity before action
            diversity_after: Diversity after action
            
        Returns:
            Reward value
        """
        # Reward improvement in fitness
        fitness_improvement = (fitness_after - fitness_before) / (np.abs(fitness_before) + 1e-8)
        
        # Reward diversity maintenance (not too much convergence)
        diversity_change = (diversity_after - diversity_before) / (diversity_before + 1e-8)
        
        # Combined reward
        reward = 0.8 * fitness_improvement + 0.2 * (-diversity_change)
        
        return float(np.clip(reward, -1, 1))
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool = False):
        """
        Update DDQN agent with experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        self.agent.store_experience(state, action, reward, next_state, done)
        
        # Train agent periodically
        if len(self.agent.replay_buffer.buffer) % 100 == 0:
            self.agent.train(batch_size=32)
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current control parameters.
        
        Returns:
            Parameters dictionary
        """
        return self.current_parameters.copy()
