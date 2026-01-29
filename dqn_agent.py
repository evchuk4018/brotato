"""
Deep Q-Network (DQN) Trading Agent - Tabula Rasa Implementation

A reinforcement learning agent for stock trading that learns from scratch
using experience replay and epsilon-greedy exploration.

Author: Brotato Trading Bot
"""

import random
from collections import deque
from typing import Tuple, List, Optional, Deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


# Type aliases for clarity
State = np.ndarray  # Shape: (60, 5) - 60 minutes of OHLCV data
Action = int  # 0: BUY, 1: HOLD, 2: SELL
Experience = Tuple[State, Action, float, State, bool]


class ActionSpace:
    """Defines the action space for the trading agent."""
    BUY: int = 0
    HOLD: int = 1
    SELL: int = 2
    SIZE: int = 3
    
    @classmethod
    def to_string(cls, action: int) -> str:
        """Convert action index to human-readable string."""
        mapping = {cls.BUY: "BUY", cls.HOLD: "HOLD", cls.SELL: "SELL"}
        return mapping.get(action, "UNKNOWN")


def build_agent(
    state_shape: Tuple[int, int] = (60, 5),
    action_size: int = 3,
    learning_rate: float = 0.001,
    lstm_units: int = 50,
    dropout_rate: float = 0.2
) -> Model:
    """
    Build a Deep Q-Network model for stock trading from scratch with random weights.
    
    The architecture consists of:
    - 2x LSTM layers with 50 units each for temporal pattern recognition
    - Dropout layers for regularization
    - Dense output layer with linear activation for Q-value estimation
    
    Args:
        state_shape: Shape of input state (sequence_length, features).
                     Default: (60, 5) for 60 minutes of OHLCV data.
        action_size: Number of possible actions. Default: 3 (BUY, HOLD, SELL).
        learning_rate: Learning rate for Adam optimizer. Default: 0.001.
        lstm_units: Number of units in each LSTM layer. Default: 50.
        dropout_rate: Dropout rate for regularization. Default: 0.2.
    
    Returns:
        Compiled Keras Model ready for training.
    
    Example:
        >>> model = build_agent()
        >>> model.summary()
    """
    model = Sequential([
        # Input layer - explicitly define input shape
        Input(shape=state_shape),
        
        # First LSTM layer - returns sequences for stacking
        LSTM(
            units=lstm_units,
            return_sequences=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            name='lstm_1'
        ),
        Dropout(rate=dropout_rate, name='dropout_1'),
        
        # Second LSTM layer - returns final hidden state
        LSTM(
            units=lstm_units,
            return_sequences=False,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            name='lstm_2'
        ),
        Dropout(rate=dropout_rate, name='dropout_2'),
        
        # Output layer - Q-values for each action
        # Linear activation for unbounded Q-value estimation
        Dense(
            units=action_size,
            activation='linear',
            kernel_initializer='glorot_uniform',
            name='q_values'
        )
    ], name='DQN_Trading_Agent')
    
    # Compile with Adam optimizer and MSE loss for Q-learning
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN training.
    
    Stores past experiences and provides random sampling for training,
    which breaks correlation between consecutive samples and stabilizes learning.
    
    Attributes:
        buffer: Deque containing experience tuples.
        maxlen: Maximum buffer capacity.
    """
    
    def __init__(self, maxlen: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            maxlen: Maximum number of experiences to store.
                   Older experiences are discarded when full.
        """
        self.buffer: Deque[Experience] = deque(maxlen=maxlen)
        self.maxlen = maxlen
    
    def add(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool
    ) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state observation.
            action: Action taken in the state.
            reward: Reward received after taking the action.
            next_state: Resulting state after the action.
            done: Whether the episode terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Randomly sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample.
        
        Returns:
            List of experience tuples.
        
        Raises:
            ValueError: If batch_size exceeds buffer length.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Batch size ({batch_size}) exceeds buffer length ({len(self.buffer)})"
            )
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self) -> int:
        """Return the current number of experiences in the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size


class DQNAgent:
    """
    Deep Q-Network Agent for Stock Trading.
    
    Implements a DQN agent with:
    - Epsilon-greedy exploration strategy with decay
    - Experience replay for stable learning
    - Target network for stable Q-value estimation
    - Double DQN architecture
    
    Attributes:
        state_shape: Shape of the state space (sequence_length, features).
        action_size: Number of possible actions.
        gamma: Discount factor for future rewards.
        epsilon: Current exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Decay rate for epsilon after each replay.
        model: Main Q-network for action selection.
        target_model: Target Q-network for stable Q-value estimation.
        memory: Experience replay buffer.
    
    Example:
        >>> agent = DQNAgent()
        >>> state = np.random.randn(60, 5)  # 60 minutes of OHLCV
        >>> action = agent.act(state)
        >>> print(f"Selected action: {ActionSpace.to_string(action)}")
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int] = (60, 5),
        action_size: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32
    ):
        """
        Initialize the DQN Agent with random weights (Tabula Rasa).
        
        Args:
            state_shape: Shape of input state (sequence_length, features).
            action_size: Number of possible actions (BUY, HOLD, SELL).
            learning_rate: Learning rate for the Adam optimizer.
            gamma: Discount factor for future rewards (0-1).
            epsilon: Initial exploration rate (0-1).
            epsilon_min: Minimum exploration rate.
            epsilon_decay: Multiplicative decay factor for epsilon.
            memory_size: Maximum size of the replay buffer.
            batch_size: Number of samples per training batch.
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Reinforcement learning hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Build main and target networks from scratch (Tabula Rasa)
        self.model = build_agent(
            state_shape=state_shape,
            action_size=action_size,
            learning_rate=learning_rate
        )
        self.target_model = build_agent(
            state_shape=state_shape,
            action_size=action_size,
            learning_rate=learning_rate
        )
        
        # Initialize target model with same weights as main model
        self.update_target()
        
        # Experience replay buffer
        self.memory = ReplayBuffer(maxlen=memory_size)
        
        # Training statistics
        self.training_step = 0
        self.total_reward = 0.0
    
    def act(self, state: State, training: bool = True) -> Action:
        """
        Select an action using epsilon-greedy policy.
        
        During training, the agent explores with probability epsilon
        and exploits (chooses best action) with probability (1 - epsilon).
        During inference, always exploits.
        
        Args:
            state: Current state observation of shape (60, 5).
            training: If True, use epsilon-greedy; if False, always exploit.
        
        Returns:
            Selected action index (0: BUY, 1: HOLD, 2: SELL).
        
        Example:
            >>> state = np.random.randn(60, 5)
            >>> action = agent.act(state, training=True)
        """
        # Exploration: random action
        if training and np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: best action based on Q-values
        # Reshape state for model: (60, 5) -> (1, 60, 5)
        state_batch = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state_batch, verbose=0)[0]
        return int(np.argmax(q_values))
    
    def remember(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool
    ) -> None:
        """
        Store an experience in the replay buffer.
        
        Args:
            state: Current state observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next state observation.
            done: Whether the episode ended.
        
        Example:
            >>> agent.remember(state, action=0, reward=10.5, next_state=new_state, done=False)
        """
        self.memory.add(state, action, reward, next_state, done)
        self.total_reward += reward
    
    def replay(self, batch_size: Optional[int] = None) -> Optional[float]:
        """
        Train the model on a batch of experiences from the replay buffer.
        
        Implements the core DQN training loop:
        1. Sample random batch from replay buffer
        2. Calculate target Q-values using target network
        3. Update main network using MSE loss
        4. Decay epsilon for exploration/exploitation balance
        
        Args:
            batch_size: Number of samples to train on.
                       Uses default batch_size if not specified.
        
        Returns:
            Training loss if training occurred, None if buffer too small.
        
        Example:
            >>> loss = agent.replay()
            >>> print(f"Training loss: {loss:.4f}")
        """
        batch_size = batch_size or self.batch_size
        
        # Check if we have enough experiences
        if not self.memory.is_ready(batch_size):
            return None
        
        # Sample random batch
        batch = self.memory.sample(batch_size)
        
        # Prepare batch arrays
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Current Q-values from main network
        current_q_values = self.model.predict(states, verbose=0)
        
        # Next Q-values from target network (for stability)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Double DQN: use main network to select action, target network for Q-value
        next_actions = np.argmax(
            self.model.predict(next_states, verbose=0), axis=1
        )
        
        # Calculate target Q-values using Bellman equation
        targets = current_q_values.copy()
        for i in range(batch_size):
            if dones[i]:
                # Terminal state: Q(s,a) = reward
                targets[i, actions[i]] = rewards[i]
            else:
                # Non-terminal: Q(s,a) = reward + gamma * max_a' Q(s',a')
                # Double DQN variant: use action from main network
                targets[i, actions[i]] = (
                    rewards[i] + self.gamma * next_q_values[i, next_actions[i]]
                )
        
        # Train the model
        history = self.model.fit(
            states,
            targets,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )
        
        # Decay epsilon
        self._decay_epsilon()
        
        self.training_step += 1
        
        return history.history['loss'][0]
    
    def update_target(self) -> None:
        """
        Update target network weights from main network.
        
        The target network provides stable Q-value estimates during training.
        This should be called periodically (e.g., every N training steps).
        
        Example:
            >>> # Update target network every 100 steps
            >>> if agent.training_step % 100 == 0:
            ...     agent.update_target()
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def soft_update_target(self, tau: float = 0.001) -> None:
        """
        Soft update target network weights (Polyak averaging).
        
        Gradually blends target network weights with main network weights.
        This provides smoother updates than hard replacement.
        
        Args:
            tau: Blending factor (0-1). Lower values = slower updates.
        
        Formula: target_weights = tau * main_weights + (1 - tau) * target_weights
        """
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        new_weights = [
            tau * mw + (1 - tau) * tw
            for mw, tw in zip(main_weights, target_weights)
        ]
        
        self.target_model.set_weights(new_weights)
    
    def _decay_epsilon(self) -> None:
        """Apply epsilon decay after training step."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def get_epsilon(self) -> float:
        """Get current exploration rate."""
        return self.epsilon
    
    def set_epsilon(self, epsilon: float) -> None:
        """
        Manually set exploration rate.
        
        Args:
            epsilon: New exploration rate (clamped to [epsilon_min, 1.0]).
        """
        self.epsilon = max(self.epsilon_min, min(1.0, epsilon))
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model (e.g., 'dqn_agent.keras').
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def get_q_values(self, state: State) -> np.ndarray:
        """
        Get Q-values for all actions given a state.
        
        Useful for debugging and visualization.
        
        Args:
            state: Current state observation.
        
        Returns:
            Array of Q-values for each action [BUY, HOLD, SELL].
        """
        state_batch = np.expand_dims(state, axis=0)
        return self.model.predict(state_batch, verbose=0)[0]
    
    def summary(self) -> None:
        """Print model architecture summary."""
        print("\n" + "=" * 60)
        print("DQN Trading Agent - Model Summary")
        print("=" * 60)
        self.model.summary()
        print(f"\nHyperparameters:")
        print(f"  - Gamma (discount): {self.gamma}")
        print(f"  - Epsilon (current): {self.epsilon:.4f}")
        print(f"  - Epsilon min: {self.epsilon_min}")
        print(f"  - Epsilon decay: {self.epsilon_decay}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Memory size: {self.memory.maxlen}")
        print(f"  - Training steps: {self.training_step}")
        print("=" * 60 + "\n")


def calculate_reward(
    action: Action,
    price_change: float,
    position: int,
    transaction_cost: float = 0.001
) -> float:
    """
    Calculate reward for a trading action.
    
    Implements a reward function that encourages profitable trades
    and penalizes poor decisions.
    
    Args:
        action: The action taken (BUY, HOLD, SELL).
        price_change: Percentage change in price (e.g., 0.02 for 2% increase).
        position: Current position (1: long, 0: neutral, -1: short).
        transaction_cost: Cost per transaction as fraction (default: 0.1%).
    
    Returns:
        Calculated reward value.
    """
    reward = 0.0
    
    if action == ActionSpace.BUY:
        if position == 0:  # Opening long position
            reward = -transaction_cost  # Cost to enter
        elif position == 1:  # Already long
            reward = -0.01  # Penalty for redundant action
    
    elif action == ActionSpace.SELL:
        if position == 1:  # Closing long position
            reward = price_change - transaction_cost
        elif position == 0:  # No position to sell
            reward = -0.01  # Penalty for invalid action
    
    elif action == ActionSpace.HOLD:
        if position == 1:  # Holding long position
            reward = price_change  # Gain/loss from holding
        else:
            reward = 0.0  # Neutral for no position
    
    return reward


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("DQN Trading Agent - Tabula Rasa Test")
    print("=" * 60)
    
    # Create agent from scratch
    agent = DQNAgent(
        state_shape=(60, 5),
        action_size=3,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32
    )
    
    # Print model summary
    agent.summary()
    
    # Simulate some experiences
    print("Simulating trading experiences...")
    for episode in range(5):
        state = np.random.randn(60, 5).astype(np.float32)
        
        for step in range(100):
            # Select action
            action = agent.act(state, training=True)
            
            # Simulate environment response
            next_state = np.random.randn(60, 5).astype(np.float32)
            reward = np.random.randn() * 0.1  # Random reward
            done = step == 99
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
        
        print(f"  Episode {episode + 1}: Buffer size = {len(agent.memory)}")
    
    # Train on experiences
    print("\nTraining on replay buffer...")
    for i in range(10):
        loss = agent.replay()
        if loss is not None:
            print(f"  Training step {i + 1}: Loss = {loss:.6f}, "
                  f"Epsilon = {agent.epsilon:.4f}")
    
    # Update target network
    agent.update_target()
    print("\nTarget network updated.")
    
    # Test inference
    print("\nTesting inference mode...")
    test_state = np.random.randn(60, 5).astype(np.float32)
    q_values = agent.get_q_values(test_state)
    action = agent.act(test_state, training=False)
    
    print(f"  Q-values: BUY={q_values[0]:.4f}, HOLD={q_values[1]:.4f}, "
          f"SELL={q_values[2]:.4f}")
    print(f"  Selected action: {ActionSpace.to_string(action)}")
    
    print("\n" + "=" * 60)
    print("DQN Agent initialization successful!")
    print("=" * 60)
