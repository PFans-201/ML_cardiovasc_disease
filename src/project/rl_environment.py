"""
Reinforcement Learning utilities for cardiovascular disease treatment optimization.

This module provides utilities for modeling treatment decisions as RL problems
as required in Milestone M6.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import random
import warnings
warnings.filterwarnings('ignore')


class CVDTreatmentEnvironment:
    """
    Reinforcement Learning environment for cardiovascular disease treatment.
    
    Models patient-treatment interactions where:
    - States: Patient condition (symptoms, risk factors, disease stage)
    - Actions: Treatment options (medications, procedures, lifestyle)
    - Rewards: Based on patient outcomes and treatment utility
    """
    
    def __init__(self, df, state_features=None, random_state=42):
        """
        Initialize CVD treatment environment.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Patient data with treatments and outcomes
        state_features : list, optional
            Features to include in state representation
        random_state : int, default=42
            Random state for reproducible results
        """
        self.df = df.copy()
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define state and action spaces
        self.state_features = state_features or [
            'age', 'gender', 'cholesterol', 'blood_pressure', 
            'risk_score', 'chest_pain', 'fatigue'
        ]
        
        # Initialize environment components
        self._setup_state_space()
        self._setup_action_space()
        self._setup_reward_function()
        
        # Current episode state
        self.current_patient = None
        self.current_state = None
        self.episode_step = 0
        self.max_episode_length = 10
        
    def _setup_state_space(self):
        """Setup state space representation."""
        # Filter available features
        available_features = [f for f in self.state_features if f in self.df.columns]
        self.state_features = available_features
        
        # Discretize continuous state features
        self.state_bins = {}
        self.state_values = {}
        
        for feature in self.state_features:
            if self.df[feature].dtype in ['float64', 'int64']:
                # Create bins for continuous features
                self.state_bins[feature] = pd.qcut(
                    self.df[feature].dropna(), 
                    q=3, 
                    labels=['Low', 'Medium', 'High'],
                    duplicates='drop'
                )
                self.state_values[feature] = ['Low', 'Medium', 'High']
            else:
                # Use existing categories for discrete features
                self.state_values[feature] = sorted(self.df[feature].dropna().unique())
        
        # Calculate state space size
        self.state_space_size = 1
        for feature in self.state_features:
            self.state_space_size *= len(self.state_values[feature])
        
        print(f"State space size: {self.state_space_size}")
    
    def _setup_action_space(self):
        """Setup action space (treatment options)."""
        if 'treatment' in self.df.columns:
            self.actions = sorted(self.df['treatment'].dropna().unique())
        else:
            # Default treatment options
            self.actions = ['no_treatment', 'medication', 'lifestyle_change', 'surgery']
        
        self.action_space_size = len(self.actions)
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        
        print(f"Action space: {self.actions}")
    
    def _setup_reward_function(self):
        """Setup reward function based on treatment outcomes."""
        # Define reward components
        self.reward_components = {
            'utility_improvement': 1.0,
            'risk_reduction': 0.5,
            'treatment_cost': -0.1,
            'side_effects': -0.3
        }
        
        # Analyze treatment outcomes from data if available
        if 'utility' in self.df.columns:
            self.utility_stats = self.df.groupby('treatment')['utility'].agg(['mean', 'std'])
        
    def reset(self, patient_id=None):
        """
        Reset environment for new episode.
        
        Parameters:
        -----------
        patient_id : str, optional
            Specific patient to simulate (if None, random selection)
            
        Returns:
        --------
        state : tuple
            Initial state representation
        """
        # Select patient for episode
        if patient_id is not None:
            patient_data = self.df[self.df.get('patient_id', 'id') == patient_id]
            if len(patient_data) > 0:
                self.current_patient = patient_data.iloc[0]
            else:
                self.current_patient = self.df.sample(1, random_state=self.random_state).iloc[0]
        else:
            self.current_patient = self.df.sample(1, random_state=self.random_state).iloc[0]
        
        # Extract initial state
        self.current_state = self._extract_state(self.current_patient)
        self.episode_step = 0
        
        return self.current_state
    
    def _extract_state(self, patient_data):
        """
        Extract state representation from patient data.
        
        Parameters:
        -----------
        patient_data : pd.Series
            Patient information
            
        Returns:
        --------
        state : tuple
            State representation
        """
        state = []
        for feature in self.state_features:
            if feature in patient_data:
                value = patient_data[feature]
                
                # Discretize if necessary
                if feature in self.state_bins:
                    # Find bin for continuous value
                    if pd.isna(value):
                        state_val = 'Medium'  # Default for missing values
                    else:
                        # Simple binning logic
                        feature_values = self.df[feature].dropna()
                        low_thresh = feature_values.quantile(0.33)
                        high_thresh = feature_values.quantile(0.67)
                        
                        if value <= low_thresh:
                            state_val = 'Low'
                        elif value <= high_thresh:
                            state_val = 'Medium'
                        else:
                            state_val = 'High'
                else:
                    state_val = str(value)
                
                state.append(state_val)
            else:
                state.append('Unknown')
        
        return tuple(state)
    
    def step(self, action):
        """
        Execute action in environment.
        
        Parameters:
        -----------
        action : int or str
            Action to take (treatment option)
            
        Returns:
        --------
        next_state : tuple
            Next state after action
        reward : float
            Reward for taking action
        done : bool
            Whether episode is finished
        info : dict
            Additional information
        """
        if isinstance(action, int):
            action_name = self.actions[action]
        else:
            action_name = action
        
        # Calculate reward
        reward = self._calculate_reward(self.current_state, action_name)
        
        # Simulate state transition
        next_state = self._simulate_transition(self.current_state, action_name)
        
        # Update episode
        self.episode_step += 1
        done = self.episode_step >= self.max_episode_length
        
        # Additional info
        info = {
            'action_taken': action_name,
            'episode_step': self.episode_step,
            'patient_id': self.current_patient.get('patient_id', 'unknown')
        }
        
        self.current_state = next_state
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, state, action):
        """
        Calculate reward for state-action pair.
        
        Parameters:
        -----------
        state : tuple
            Current state
        action : str
            Action taken
            
        Returns:
        --------
        reward : float
            Calculated reward
        """
        reward = 0.0
        
        # Extract risk level from state
        risk_level = 'Medium'  # Default
        if 'risk_score' in self.state_features:
            risk_idx = self.state_features.index('risk_score')
            if risk_idx < len(state):
                risk_level = state[risk_idx]
        
        # Base reward based on treatment appropriateness
        treatment_rewards = {
            'no_treatment': {'Low': 0.5, 'Medium': -0.2, 'High': -1.0},
            'medication': {'Low': -0.1, 'Medium': 0.8, 'High': 0.6},
            'lifestyle_change': {'Low': 0.3, 'Medium': 0.5, 'High': 0.2},
            'surgery': {'Low': -0.8, 'Medium': -0.2, 'High': 1.0}
        }
        
        if action in treatment_rewards and risk_level in treatment_rewards[action]:
            reward += treatment_rewards[action][risk_level]
        
        # Penalty for inappropriate treatment combinations
        if action == 'surgery' and risk_level == 'Low':
            reward -= 1.0  # Major penalty for unnecessary surgery
        
        # Bonus for early intervention
        if action in ['medication', 'lifestyle_change'] and risk_level == 'Medium':
            reward += 0.3
        
        return reward
    
    def _simulate_transition(self, state, action):
        """
        Simulate state transition based on action.
        
        Parameters:
        -----------
        state : tuple
            Current state
        action : str
            Action taken
            
        Returns:
        --------
        next_state : tuple
            Next state
        """
        next_state = list(state)
        
        # Simulate treatment effects
        if 'risk_score' in self.state_features:
            risk_idx = self.state_features.index('risk_score')
            if risk_idx < len(next_state):
                current_risk = next_state[risk_idx]
                
                # Treatment effects on risk
                if action == 'medication':
                    if current_risk == 'High' and np.random.random() < 0.7:
                        next_state[risk_idx] = 'Medium'
                    elif current_risk == 'Medium' and np.random.random() < 0.5:
                        next_state[risk_idx] = 'Low'
                
                elif action == 'surgery':
                    if current_risk == 'High' and np.random.random() < 0.9:
                        next_state[risk_idx] = 'Low'
                
                elif action == 'lifestyle_change':
                    if current_risk == 'Medium' and np.random.random() < 0.4:
                        next_state[risk_idx] = 'Low'
        
        # Simulate symptom changes
        symptom_features = ['chest_pain', 'fatigue']
        for symptom in symptom_features:
            if symptom in self.state_features:
                symptom_idx = self.state_features.index(symptom)
                if symptom_idx < len(next_state):
                    # Treatment can reduce symptoms
                    if action in ['medication', 'surgery'] and np.random.random() < 0.6:
                        if next_state[symptom_idx] == 'Yes':
                            next_state[symptom_idx] = 'No'
        
        return tuple(next_state)
    
    def get_state_index(self, state):
        """Convert state tuple to index for tabular methods."""
        index = 0
        multiplier = 1
        
        for i in reversed(range(len(state))):
            feature = self.state_features[i]
            value = state[i]
            
            if value in self.state_values[feature]:
                value_idx = self.state_values[feature].index(value)
            else:
                value_idx = 0  # Default
            
            index += value_idx * multiplier
            multiplier *= len(self.state_values[feature])
        
        return index


class CVDQLearning:
    """
    Q-Learning algorithm for cardiovascular disease treatment optimization.
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995):
        """
        Initialize Q-Learning agent.
        
        Parameters:
        -----------
        env : CVDTreatmentEnvironment
            Environment to learn in
        learning_rate : float, default=0.1
            Learning rate for Q-value updates
        discount_factor : float, default=0.95
            Discount factor for future rewards
        epsilon : float, default=0.1
            Exploration rate for epsilon-greedy policy
        epsilon_decay : float, default=0.995
            Decay rate for epsilon
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table
        self.q_table = defaultdict(lambda: np.zeros(env.action_space_size))
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [],
            'q_value_changes': []
        }
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Parameters:
        -----------
        state : tuple
            Current state
        training : bool, default=True
            Whether in training mode (affects exploration)
            
        Returns:
        --------
        action : int
            Selected action index
        """
        state_index = self.env.get_state_index(state)
        
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.env.action_space_size)
        else:
            # Exploit: best action according to Q-table
            return np.argmax(self.q_table[state_index])
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning update rule.
        
        Parameters:
        -----------
        state : tuple
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : tuple
            Next state
        done : bool
            Whether episode is finished
        """
        state_index = self.env.get_state_index(state)
        next_state_index = self.env.get_state_index(next_state)
        
        current_q = self.q_table[state_index][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_index])
        
        # Q-learning update
        self.q_table[state_index][action] += self.learning_rate * (target_q - current_q)
        
        # Track Q-value change
        q_change = abs(self.q_table[state_index][action] - current_q)
        self.training_history['q_value_changes'].append(q_change)
    
    def train(self, n_episodes=1000, verbose=True):
        """
        Train Q-learning agent.
        
        Parameters:
        -----------
        n_episodes : int, default=1000
            Number of training episodes
        verbose : bool, default=True
            Whether to print training progress
            
        Returns:
        --------
        training_history : dict
            Training metrics and history
        """
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select and execute action
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state, done)
                
                # Update episode metrics
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
            
            # Record training metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['epsilon_values'].append(self.epsilon)
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return self.training_history
    
    def evaluate_policy(self, n_episodes=100):
        """
        Evaluate learned policy.
        
        Parameters:
        -----------
        n_episodes : int, default=100
            Number of evaluation episodes
            
        Returns:
        --------
        evaluation_results : dict
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.select_action(state, training=False)
                next_state, reward, done, _ = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        evaluation_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        return evaluation_results
    
    def get_policy(self):
        """
        Extract learned policy from Q-table.
        
        Returns:
        --------
        policy : dict
            Mapping from state to best action
        """
        policy = {}
        
        for state_index, q_values in self.q_table.items():
            best_action = np.argmax(q_values)
            policy[state_index] = {
                'action': best_action,
                'action_name': self.env.actions[best_action],
                'q_values': q_values.copy()
            }
        
        return policy


class CVDPolicyGradient:
    """
    REINFORCE Policy Gradient algorithm for treatment optimization.
    """
    
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99):
        """
        Initialize Policy Gradient agent.
        
        Parameters:
        -----------
        env : CVDTreatmentEnvironment
            Environment to learn in
        learning_rate : float, default=0.01
            Learning rate for policy updates
        discount_factor : float, default=0.99
            Discount factor for future rewards
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Simple linear policy (state features -> action probabilities)
        self.policy_weights = np.random.normal(0, 0.1, 
                                             (len(env.state_features), env.action_space_size))
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'policy_entropy': [],
            'loss_values': []
        }
    
    def get_action_probabilities(self, state):
        """
        Get action probabilities for given state.
        
        Parameters:
        -----------
        state : tuple
            Current state
            
        Returns:
        --------
        probabilities : np.array
            Action probabilities
        """
        # Convert state to feature vector
        state_vector = self._state_to_vector(state)
        
        # Compute logits
        logits = np.dot(state_vector, self.policy_weights)
        
        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        return probabilities
    
    def select_action(self, state):
        """
        Select action based on policy probabilities.
        
        Parameters:
        -----------
        state : tuple
            Current state
            
        Returns:
        --------
        action : int
            Selected action
        action_prob : float
            Probability of selected action
        """
        probabilities = self.get_action_probabilities(state)
        action = np.random.choice(len(probabilities), p=probabilities)
        
        return action, probabilities[action]
    
    def _state_to_vector(self, state):
        """
        Convert state tuple to numerical vector.
        
        Parameters:
        -----------
        state : tuple
            State representation
            
        Returns:
        --------
        vector : np.array
            Numerical state vector
        """
        vector = np.zeros(len(self.env.state_features))
        
        for i, feature in enumerate(self.env.state_features):
            if i < len(state) and feature in self.env.state_values:
                value = state[i]
                if value in self.env.state_values[feature]:
                    # One-hot encoding for categorical features
                    value_idx = self.env.state_values[feature].index(value)
                    vector[i] = value_idx / len(self.env.state_values[feature])
                else:
                    vector[i] = 0.5  # Default for unknown values
        
        return vector
    
    def train(self, n_episodes=1000, verbose=True):
        """
        Train policy gradient agent using REINFORCE.
        
        Parameters:
        -----------
        n_episodes : int, default=1000
            Number of training episodes
        verbose : bool, default=True
            Whether to print training progress
            
        Returns:
        --------
        training_history : dict
            Training metrics and history
        """
        for episode in range(n_episodes):
            # Collect episode trajectory
            states, actions, rewards, action_probs = self._collect_episode()
            
            # Calculate discounted returns
            returns = self._calculate_returns(rewards)
            
            # Update policy
            loss = self._update_policy(states, actions, returns, action_probs)
            
            # Record training metrics
            episode_reward = sum(rewards)
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['loss_values'].append(loss)
            
            # Calculate policy entropy (exploration measure)
            avg_entropy = np.mean([self._calculate_entropy(self.get_action_probabilities(s)) 
                                 for s in states])
            self.training_history['policy_entropy'].append(avg_entropy)
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Policy Entropy: {avg_entropy:.3f}")
        
        return self.training_history
    
    def _collect_episode(self):
        """Collect full episode trajectory."""
        states, actions, rewards, action_probs = [], [], [], []
        
        state = self.env.reset()
        
        while True:
            action, action_prob = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            action_probs.append(action_prob)
            
            state = next_state
            
            if done:
                break
        
        return states, actions, rewards, action_probs
    
    def _calculate_returns(self, rewards):
        """Calculate discounted returns."""
        returns = []
        discounted_return = 0
        
        for reward in reversed(rewards):
            discounted_return = reward + self.discount_factor * discounted_return
            returns.insert(0, discounted_return)
        
        # Normalize returns
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        return returns
    
    def _update_policy(self, states, actions, returns, action_probs):
        """Update policy using REINFORCE algorithm."""
        policy_gradient = np.zeros_like(self.policy_weights)
        
        for i, (state, action, g, prob) in enumerate(zip(states, actions, returns, action_probs)):
            state_vector = self._state_to_vector(state)
            
            # Calculate gradient for this step
            grad = np.outer(state_vector, np.zeros(self.env.action_space_size))
            grad[:, action] = state_vector / prob
            
            # Weight by return
            policy_gradient += g * grad
        
        # Update policy weights
        self.policy_weights += self.learning_rate * policy_gradient / len(states)
        
        # Return loss (negative log probability weighted by return)
        loss = -np.mean([np.log(prob + 1e-8) * g for prob, g in zip(action_probs, returns)])
        
        return loss
    
    def _calculate_entropy(self, probabilities):
        """Calculate entropy of action probabilities."""
        return -np.sum(probabilities * np.log(probabilities + 1e-8))


def visualize_training_results(agent, save_path=None, figsize=(15, 10)):
    """
    Visualize training results for RL agents.
    
    Parameters:
    -----------
    agent : CVDQLearning or CVDPolicyGradient
        Trained RL agent
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(15, 10)
        Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    history = agent.training_history
    
    # 1. Episode rewards
    axes[0, 0].plot(history['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Moving average of rewards
    window_size = min(100, len(history['episode_rewards']) // 10)
    if window_size > 1:
        moving_avg = np.convolve(history['episode_rewards'], 
                               np.ones(window_size)/window_size, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window_size})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Episode lengths
    if 'episode_lengths' in history:
        axes[0, 2].plot(history['episode_lengths'])
        axes[0, 2].set_title('Episode Lengths')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Steps')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Algorithm-specific plots
    if hasattr(agent, 'epsilon'):  # Q-Learning
        if 'epsilon_values' in history:
            axes[1, 0].plot(history['epsilon_values'])
            axes[1, 0].set_title('Epsilon Decay')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Epsilon')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'q_value_changes' in history:
            axes[1, 1].plot(history['q_value_changes'])
            axes[1, 1].set_title('Q-Value Changes')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('|ΔQ|')
            axes[1, 1].grid(True, alpha=0.3)
    
    else:  # Policy Gradient
        if 'policy_entropy' in history:
            axes[1, 0].plot(history['policy_entropy'])
            axes[1, 0].set_title('Policy Entropy')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'loss_values' in history:
            axes[1, 1].plot(history['loss_values'])
            axes[1, 1].set_title('Policy Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Reward distribution
    axes[1, 2].hist(history['episode_rewards'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Reward Distribution')
    axes[1, 2].set_xlabel('Episode Reward')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_rl_algorithms(q_agent, pg_agent, save_path=None):
    """
    Compare performance of Q-Learning and Policy Gradient agents.
    
    Parameters:
    -----------
    q_agent : CVDQLearning
        Trained Q-Learning agent
    pg_agent : CVDPolicyGradient
        Trained Policy Gradient agent
    save_path : str, optional
        Path to save comparison figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Training curves comparison
    q_rewards = q_agent.training_history['episode_rewards']
    pg_rewards = pg_agent.training_history['episode_rewards']
    
    axes[0].plot(q_rewards, label='Q-Learning', alpha=0.7)
    axes[0].plot(pg_rewards, label='Policy Gradient', alpha=0.7)
    axes[0].set_title('Training Performance Comparison')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Final performance evaluation
    q_eval = q_agent.evaluate_policy(100)
    pg_eval = pg_agent.evaluate_policy(100) if hasattr(pg_agent, 'evaluate_policy') else None
    
    if pg_eval:
        algorithms = ['Q-Learning', 'Policy Gradient']
        mean_rewards = [q_eval['mean_reward'], pg_eval['mean_reward']]
        std_rewards = [q_eval['std_reward'], pg_eval['std_reward']]
        
        axes[1].bar(algorithms, mean_rewards, yerr=std_rewards, capsize=5)
        axes[1].set_title('Final Performance Comparison')
        axes[1].set_ylabel('Mean Episode Reward')
        axes[1].grid(True, alpha=0.3)
    
    # 3. Learning efficiency (time to convergence)
    window_size = 50
    q_smooth = np.convolve(q_rewards, np.ones(window_size)/window_size, mode='valid')
    pg_smooth = np.convolve(pg_rewards, np.ones(window_size)/window_size, mode='valid')
    
    axes[2].plot(q_smooth, label='Q-Learning (smoothed)')
    axes[2].plot(pg_smooth, label='Policy Gradient (smoothed)')
    axes[2].set_title('Learning Curves (Smoothed)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Average Reward')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()