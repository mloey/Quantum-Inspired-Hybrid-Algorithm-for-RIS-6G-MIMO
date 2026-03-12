"""
Double Deep Q-Network (DDQN) for reinforcement learning in optimization.
"""

import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Prioritized Experience Replay buffer for DDQN training.
    """
    
    def __init__(self, buffer_size: int = 10000, alpha: float = 0.6):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum buffer size
            alpha: Priority exponent (0 = no prioritization, 1 = full prioritization)
        """
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(buffer_size)
        self.position = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        Add experience to buffer with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        max_priority = self.priorities[:len(self.buffer)].max()
        if max_priority == 0:
            max_priority = 1.0
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, List[int], np.ndarray, 
                                                np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch using prioritized sampling.
        
        Args:
            batch_size: Batch size
            
        Returns:
            States, actions, rewards, next_states, dones, weights
        """
        buffer_len = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:buffer_len]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (buffer_len * probs[indices]) ** (-0.4)  # Beta = 0.4
        weights = weights / weights.max()
        
        # Extract experiences
        states, actions, rewards = [], [], []
        next_states, dones = [], []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (np.array(states), actions, np.array(rewards),
                np.array(next_states), np.array(dones), weights)
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities based on temporal difference errors.
        
        Args:
            indices: Experience indices
            td_errors: TD errors for prioritization
        """
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (np.abs(td_error) + 1e-6) ** self.alpha


class DDQNAgent:
    """
    Double Deep Q-Network agent for adaptive parameter control in optimization.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001, discount_factor: float = 0.99):
        """
        Initialize DDQN agent.
        
        Args:
            state_size: State space dimensionality
            action_size: Number of discrete actions
            learning_rate: Learning rate
            discount_factor: Discount factor for future rewards
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        
        # Q-network weights (simple linear approximation)
        self.q_network_weights = np.random.randn(state_size, action_size) * 0.01
        self.target_network_weights = self.q_network_weights.copy()
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=10000)
        
        # Training parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_freq = 10
        self.train_step = 0
        
        logger.info(f"Initialized DDQN agent with state_size={state_size}, action_size={action_size}")
    
    def _forward(self, state: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Calculate Q-values using linear network.
        
        Args:
            state: Input state
            weights: Network weights
            
        Returns:
            Q-values for all actions
        """
        # Simple linear transformation
        q_values = np.dot(state, weights)
        return q_values
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            q_values = self._forward(state, self.q_network_weights)
            return np.argmax(q_values)
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode flag
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self, batch_size: int = 32) -> float:
        """
        Train DDQN using batch from replay buffer.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Mean TD error
        """
        if len(self.replay_buffer.buffer) < batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones, weights = \
            self.replay_buffer.sample(batch_size)
        
        # Calculate target Q-values using target network (Double DQN)
        next_q_values_current = self._forward(next_states, self.q_network_weights)
        next_actions = np.argmax(next_q_values_current, axis=1)
        
        next_q_values_target = self._forward(next_states, self.target_network_weights)
        next_q_max = next_q_values_target[np.arange(batch_size), next_actions]
        
        target_q_values = rewards + self.gamma * next_q_max * (1 - dones)
        
        # Calculate current Q-values
        current_q_values = self._forward(states, self.q_network_weights)
        current_q = current_q_values[np.arange(batch_size), actions]
        
        # Calculate TD errors
        td_errors = target_q_values - current_q
        
        # Update priorities
        self.replay_buffer.update_priorities(list(range(batch_size)), td_errors)
        
        # Gradient descent step (simplified)
        learning_rate = self.learning_rate * np.mean(np.abs(weights))
        for i in range(batch_size):
            gradient = -td_errors[i] * states[i]
            self.q_network_weights[:, actions[i]] += learning_rate * gradient
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Periodically update target network
        self.train_step += 1
        if self.train_step % self.update_freq == 0:
            self.target_network_weights = self.q_network_weights.copy()
        
        return np.mean(np.abs(td_errors))
    
    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions.
        
        Args:
            state: Current state
            
        Returns:
            Q-values
        """
        return self._forward(state, self.q_network_weights)
