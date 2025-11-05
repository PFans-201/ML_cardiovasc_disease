"""
Hidden Markov Model utilities for cardiovascular disease progression analysis.

This module provides utilities for modeling disease progression using HMMs
as required in Milestone M5.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM, MultinomialHMM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class CVDHiddenMarkovModel:
    """
    Hidden Markov Model for cardiovascular disease progression.
    
    Provides comprehensive HMM functionality including:
    - Disease progression modeling
    - State sequence inference
    - Future progression prediction
    - Clinical interpretation of states
    """
    
    def __init__(self, n_states=4, random_state=42):
        """
        Initialize CVD Hidden Markov Model.
        
        Parameters:
        -----------
        n_states : int, default=4
            Number of hidden states (disease stages)
        random_state : int, default=42
            Random state for reproducible results
        """
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.state_names = None
        self.feature_names = None
        self.patient_sequences = {}
        
    def prepare_sequential_data(self, df, patient_id_col='patient_id', 
                              time_col='encounter_date', feature_cols=None):
        """
        Prepare patient data for sequential HMM modeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Patient encounter data
        patient_id_col : str, default='patient_id'
            Column name for patient identifiers
        time_col : str, default='encounter_date'
            Column name for time/date information
        feature_cols : list, optional
            Specific feature columns to use
            
        Returns:
        --------
        sequences : list
            List of patient observation sequences
        lengths : list
            List of sequence lengths for each patient
        """
        # Sort by patient and time
        df_sorted = df.sort_values([patient_id_col, time_col])
        
        # Select features for observation sequences
        if feature_cols is None:
            # Exclude ID and time columns, select numeric features
            exclude_cols = [patient_id_col, time_col]
            feature_cols = [col for col in df_sorted.columns 
                          if col not in exclude_cols and 
                          df_sorted[col].dtype in ['float64', 'int64']]
        
        self.feature_names = feature_cols
        
        # Create sequences for each patient
        sequences = []
        lengths = []
        patient_ids = []
        
        for patient_id in df_sorted[patient_id_col].unique():
            patient_data = df_sorted[df_sorted[patient_id_col] == patient_id]
            
            # Extract observation sequence for this patient
            observations = patient_data[feature_cols].values
            
            # Skip patients with insufficient data
            if len(observations) >= 2:
                sequences.append(observations)
                lengths.append(len(observations))
                patient_ids.append(patient_id)
                
                # Store patient sequence info
                self.patient_sequences[patient_id] = {
                    'observations': observations,
                    'length': len(observations),
                    'time_points': patient_data[time_col].values
                }
        
        # Concatenate all sequences for HMM training
        X = np.vstack(sequences)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, lengths, patient_ids
    
    def fit_gaussian_hmm(self, X, lengths, covariance_type='full'):
        """
        Fit Gaussian HMM to the sequential data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Concatenated observation sequences
        lengths : list
            Length of each patient sequence
        covariance_type : str, default='full'
            Type of covariance matrix ('full', 'diag', 'tied', 'spherical')
            
        Returns:
        --------
        self : CVDHiddenMarkovModel
            Fitted model
        """
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=covariance_type,
            random_state=self.random_state,
            n_iter=100
        )
        
        # Fit the model
        self.model.fit(X, lengths)
        
        # Define state names based on progression
        self.state_names = [f'Stage_{i}' for i in range(self.n_states)]
        
        return self
    
    def fit_multinomial_hmm(self, X_discrete, lengths):
        """
        Fit Multinomial HMM to discrete sequential data.
        
        Parameters:
        -----------
        X_discrete : array-like, shape (n_samples, n_features)
            Discretized observation sequences
        lengths : list
            Length of each patient sequence
            
        Returns:
        --------
        self : CVDHiddenMarkovModel
            Fitted model
        """
        # Convert to integers if needed
        if X_discrete.dtype != int:
            X_discrete = X_discrete.astype(int)
        
        self.model = MultinomialHMM(
            n_components=self.n_states,
            random_state=self.random_state,
            n_iter=100
        )
        
        # Fit the model
        self.model.fit(X_discrete, lengths)
        
        # Define state names
        self.state_names = [f'Stage_{i}' for i in range(self.n_states)]
        
        return self
    
    def predict_state_sequences(self, X=None, lengths=None, patient_id=None):
        """
        Predict most likely state sequences using Viterbi algorithm.
        
        Parameters:
        -----------
        X : array-like, optional
            Observation sequences (if None, uses training data)
        lengths : list, optional
            Sequence lengths (if None, uses training data)
        patient_id : str, optional
            Specific patient ID to predict for
            
        Returns:
        --------
        state_sequences : dict
            Dictionary mapping patient_id to state sequence
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        state_sequences = {}
        
        if patient_id is not None and patient_id in self.patient_sequences:
            # Predict for specific patient
            patient_obs = self.patient_sequences[patient_id]['observations']
            patient_obs_scaled = self.scaler.transform(patient_obs)
            
            states = self.model.predict(patient_obs_scaled)
            state_sequences[patient_id] = {
                'states': states,
                'state_names': [self.state_names[s] for s in states],
                'length': len(states)
            }
        else:
            # Predict for all patients
            start_idx = 0
            for pid, seq_info in self.patient_sequences.items():
                end_idx = start_idx + seq_info['length']
                
                if X is not None:
                    obs_seq = X[start_idx:end_idx]
                else:
                    obs_seq = self.scaler.transform(seq_info['observations'])
                
                states = self.model.predict(obs_seq)
                state_sequences[pid] = {
                    'states': states,
                    'state_names': [self.state_names[s] for s in states],
                    'length': len(states)
                }
                
                start_idx = end_idx
        
        return state_sequences
    
    def compute_state_probabilities(self, X=None, patient_id=None):
        """
        Compute state probabilities using Forward-Backward algorithm.
        
        Parameters:
        -----------
        X : array-like, optional
            Observation sequences
        patient_id : str, optional
            Specific patient ID to compute for
            
        Returns:
        --------
        state_probabilities : dict
            Dictionary mapping patient_id to state probability matrix
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        state_probabilities = {}
        
        if patient_id is not None and patient_id in self.patient_sequences:
            # Compute for specific patient
            patient_obs = self.patient_sequences[patient_id]['observations']
            patient_obs_scaled = self.scaler.transform(patient_obs)
            
            probs = self.model.predict_proba(patient_obs_scaled)
            state_probabilities[patient_id] = {
                'probabilities': probs,
                'time_points': self.patient_sequences[patient_id]['time_points']
            }
        else:
            # Compute for all patients
            for pid, seq_info in self.patient_sequences.items():
                obs_seq = self.scaler.transform(seq_info['observations'])
                probs = self.model.predict_proba(obs_seq)
                
                state_probabilities[pid] = {
                    'probabilities': probs,
                    'time_points': seq_info['time_points']
                }
        
        return state_probabilities
    
    def analyze_transitions(self):
        """
        Analyze transition probabilities between disease states.
        
        Returns:
        --------
        transition_analysis : dict
            Analysis of transition patterns
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        transition_matrix = self.model.transmat_
        
        analysis = {
            'transition_matrix': transition_matrix,
            'state_names': self.state_names,
            'stable_states': [],
            'progressive_transitions': [],
            'regressive_transitions': []
        }
        
        # Identify stable states (high self-transition probability)
        for i in range(self.n_states):
            if transition_matrix[i, i] > 0.7:  # Threshold for stability
                analysis['stable_states'].append(self.state_names[i])
        
        # Identify progressive transitions (to higher numbered states)
        for i in range(self.n_states):
            for j in range(i + 1, self.n_states):
                if transition_matrix[i, j] > 0.1:  # Threshold for significant transition
                    analysis['progressive_transitions'].append(
                        (self.state_names[i], self.state_names[j], transition_matrix[i, j])
                    )
        
        # Identify regressive transitions (to lower numbered states)
        for i in range(self.n_states):
            for j in range(i):
                if transition_matrix[i, j] > 0.1:
                    analysis['regressive_transitions'].append(
                        (self.state_names[i], self.state_names[j], transition_matrix[i, j])
                    )
        
        return analysis
    
    def predict_future_progression(self, patient_id, n_steps=5):
        """
        Predict future disease progression for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        n_steps : int, default=5
            Number of future time steps to predict
            
        Returns:
        --------
        future_predictions : dict
            Future state predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        if patient_id not in self.patient_sequences:
            raise ValueError(f"Patient {patient_id} not found in training data")
        
        # Get current state probabilities for the patient
        current_obs = self.patient_sequences[patient_id]['observations']
        current_obs_scaled = self.scaler.transform(current_obs)
        current_state_probs = self.model.predict_proba(current_obs_scaled)[-1]  # Last time point
        
        # Predict future states
        future_probs = []
        current_prob = current_state_probs
        
        for step in range(n_steps):
            # Apply transition matrix to get next state probabilities
            next_prob = current_prob @ self.model.transmat_
            future_probs.append(next_prob)
            current_prob = next_prob
        
        # Get most likely states
        most_likely_states = [np.argmax(probs) for probs in future_probs]
        most_likely_state_names = [self.state_names[state] for state in most_likely_states]
        
        future_predictions = {
            'patient_id': patient_id,
            'current_state_probs': current_state_probs,
            'future_state_probs': future_probs,
            'most_likely_states': most_likely_states,
            'most_likely_state_names': most_likely_state_names,
            'prediction_steps': n_steps
        }
        
        return future_predictions
    
    def evaluate_model(self, X, lengths):
        """
        Evaluate HMM model performance.
        
        Parameters:
        -----------
        X : array-like
            Observation sequences
        lengths : list
            Sequence lengths
            
        Returns:
        --------
        evaluation : dict
            Model evaluation metrics
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        # Calculate log-likelihood
        log_likelihood = self.model.score(X, lengths)
        
        # Calculate AIC and BIC
        n_params = self._count_parameters()
        n_samples = X.shape[0]
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_samples) * n_params - 2 * log_likelihood
        
        evaluation = {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'n_parameters': n_params,
            'n_samples': n_samples,
            'log_likelihood_per_sample': log_likelihood / n_samples
        }
        
        return evaluation
    
    def _count_parameters(self):
        """Count the number of parameters in the HMM."""
        if isinstance(self.model, GaussianHMM):
            # Transition matrix: n_states * (n_states - 1)
            # Initial state: n_states - 1
            # Means: n_states * n_features
            # Covariances: depends on covariance type
            n_features = self.model.means_.shape[1]
            
            trans_params = self.n_states * (self.n_states - 1)
            init_params = self.n_states - 1
            mean_params = self.n_states * n_features
            
            if self.model.covariance_type == 'full':
                cov_params = self.n_states * n_features * (n_features + 1) // 2
            elif self.model.covariance_type == 'diag':
                cov_params = self.n_states * n_features
            elif self.model.covariance_type == 'tied':
                cov_params = n_features * (n_features + 1) // 2
            else:  # spherical
                cov_params = self.n_states
            
            return trans_params + init_params + mean_params + cov_params
        
        return 0  # Simplified for other model types
    
    def visualize_results(self, patient_id=None, save_path=None, figsize=(15, 10)):
        """
        Visualize HMM results including state sequences and transitions.
        
        Parameters:
        -----------
        patient_id : str, optional
            Specific patient to visualize
        save_path : str, optional
            Path to save the figure
        figsize : tuple, default=(15, 10)
            Figure size
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Transition matrix heatmap
        sns.heatmap(self.model.transmat_, annot=True, cmap='Blues', 
                   xticklabels=self.state_names, yticklabels=self.state_names,
                   ax=axes[0, 0])
        axes[0, 0].set_title('Transition Matrix')
        axes[0, 0].set_xlabel('To State')
        axes[0, 0].set_ylabel('From State')
        
        # 2. Initial state distribution
        axes[0, 1].bar(self.state_names, self.model.startprob_)
        axes[0, 1].set_title('Initial State Distribution')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. State means (for Gaussian HMM)
        if hasattr(self.model, 'means_'):
            means_df = pd.DataFrame(self.model.means_, 
                                  index=self.state_names,
                                  columns=self.feature_names[:self.model.means_.shape[1]])
            sns.heatmap(means_df.T, annot=True, cmap='RdYlBu_r', ax=axes[0, 2])
            axes[0, 2].set_title('State Emission Means')
            axes[0, 2].set_xlabel('State')
        
        # 4. Example patient trajectory
        if patient_id and patient_id in self.patient_sequences:
            state_seq = self.predict_state_sequences(patient_id=patient_id)[patient_id]
            time_points = range(len(state_seq['states']))
            
            axes[1, 0].plot(time_points, state_seq['states'], 'o-', linewidth=2, markersize=8)
            axes[1, 0].set_title(f'State Sequence for Patient {patient_id}')
            axes[1, 0].set_xlabel('Time Point')
            axes[1, 0].set_ylabel('State')
            axes[1, 0].set_yticks(range(self.n_states))
            axes[1, 0].set_yticklabels(self.state_names)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. State duration analysis
        all_sequences = self.predict_state_sequences()
        state_durations = {state: [] for state in range(self.n_states)}
        
        for pid, seq_info in all_sequences.items():
            states = seq_info['states']
            current_state = states[0]
            duration = 1
            
            for i in range(1, len(states)):
                if states[i] == current_state:
                    duration += 1
                else:
                    state_durations[current_state].append(duration)
                    current_state = states[i]
                    duration = 1
            state_durations[current_state].append(duration)
        
        # Plot state durations
        duration_data = []
        for state, durations in state_durations.items():
            if durations:
                duration_data.extend([(state, d) for d in durations])
        
        if duration_data:
            duration_df = pd.DataFrame(duration_data, columns=['State', 'Duration'])
            duration_df['State_Name'] = duration_df['State'].map(
                {i: self.state_names[i] for i in range(self.n_states)}
            )
            
            sns.boxplot(data=duration_df, x='State_Name', y='Duration', ax=axes[1, 1])
            axes[1, 1].set_title('State Duration Distribution')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Model convergence (if available)
        if hasattr(self.model, 'monitor_') and hasattr(self.model.monitor_, 'history'):
            axes[1, 2].plot(self.model.monitor_.history)
            axes[1, 2].set_title('Model Convergence')
            axes[1, 2].set_xlabel('Iteration')
            axes[1, 2].set_ylabel('Log-likelihood')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def interpret_states_clinically(self, df_original=None):
        """
        Provide clinical interpretation of learned HMM states.
        
        Parameters:
        -----------
        df_original : pd.DataFrame, optional
            Original dataframe with clinical variables
            
        Returns:
        --------
        interpretations : dict
            Clinical interpretation of each state
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        interpretations = {}
        
        # Get state characteristics from emission parameters
        if hasattr(self.model, 'means_'):
            state_means = self.model.means_
            
            for i, state_name in enumerate(self.state_names):
                mean_values = state_means[i]
                
                # Identify high and low features
                feature_analysis = {}
                for j, feature in enumerate(self.feature_names[:len(mean_values)]):
                    feature_analysis[feature] = mean_values[j]
                
                # Sort features by magnitude
                sorted_features = sorted(feature_analysis.items(), 
                                       key=lambda x: abs(x[1]), reverse=True)
                
                # Clinical interpretation based on feature patterns
                interpretation = self._generate_clinical_interpretation(
                    state_name, sorted_features[:3]  # Top 3 features
                )
                
                interpretations[state_name] = {
                    'feature_values': feature_analysis,
                    'top_features': sorted_features[:3],
                    'clinical_interpretation': interpretation
                }
        
        return interpretations
    
    def _generate_clinical_interpretation(self, state_name, top_features):
        """
        Generate clinical interpretation based on feature patterns.
        
        Parameters:
        -----------
        state_name : str
            Name of the state
        top_features : list
            List of (feature_name, value) tuples
            
        Returns:
        --------
        interpretation : str
            Clinical interpretation
        """
        interpretations = {
            'risk_score': {
                'high': 'high cardiovascular risk',
                'low': 'low cardiovascular risk'
            },
            'cholesterol': {
                'high': 'elevated cholesterol levels',
                'low': 'normal cholesterol levels'
            },
            'blood_pressure': {
                'high': 'hypertension',
                'low': 'normal blood pressure'
            },
            'chest_pain': {
                'high': 'frequent chest pain symptoms',
                'low': 'minimal chest pain'
            },
            'fatigue': {
                'high': 'significant fatigue',
                'low': 'normal energy levels'
            }
        }
        
        characteristics = []
        for feature, value in top_features:
            if feature in interpretations:
                if value > 0:
                    characteristics.append(interpretations[feature]['high'])
                else:
                    characteristics.append(interpretations[feature]['low'])
        
        if characteristics:
            interpretation = f"Patients in {state_name} typically have {', '.join(characteristics)}"
        else:
            interpretation = f"Clinical interpretation for {state_name} requires domain expertise"
        
        return interpretation