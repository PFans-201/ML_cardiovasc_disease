"""
Bayesian Network utilities for cardiovascular disease analysis.

This module provides utilities for designing, learning, and performing inference
on Bayesian Networks as required in Milestones M2-M4.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import PC, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.sampling import BayesianModelSampling
import warnings
warnings.filterwarnings('ignore')


class CVDBayesianNetwork:
    """
    Bayesian Network for cardiovascular disease analysis.
    
    Provides comprehensive BN functionality including:
    - Structure design and validation
    - Parameter estimation from data
    - Exact and approximate inference
    - Clinical query analysis
    """
    
    def __init__(self, random_state=42):
        """
        Initialize CVD Bayesian Network.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducible results
        """
        self.random_state = random_state
        self.model = None
        self.variable_states = {}
        self.inference_engine = None
        
    def create_hand_designed_structure(self):
        """
        Create hand-designed BN structure based on medical knowledge.
        
        Includes minimum required variables: risk_score, state, treatment, utility
        Plus additional symptoms and lab values for evidence.
        
        Returns:
        --------
        model : BayesianNetwork
            Hand-designed network structure
        """
        # Define network structure based on medical knowledge
        edges = [
            # Risk factors influence disease state
            ('age', 'risk_score'),
            ('gender', 'risk_score'),
            ('cholesterol', 'risk_score'),
            ('blood_pressure', 'risk_score'),
            
            # Risk score influences disease state
            ('risk_score', 'state'),
            
            # Disease state influences symptoms
            ('state', 'chest_pain'),
            ('state', 'fatigue'),
            ('state', 'shortness_of_breath'),
            
            # Disease state influences treatment decisions
            ('state', 'treatment'),
            ('risk_score', 'treatment'),
            
            # Treatment and state influence utility/outcome
            ('treatment', 'utility'),
            ('state', 'utility'),
            
            # Some direct symptom influences
            ('cholesterol', 'fatigue'),
            ('blood_pressure', 'chest_pain'),
        ]
        
        self.model = BayesianNetwork(edges)
        
        # Define variable states (will be updated based on actual data)
        self.variable_states = {
            'age': ['Young', 'Middle', 'Old'],
            'gender': ['Male', 'Female'],
            'cholesterol': ['Normal', 'High'],
            'blood_pressure': ['Normal', 'High'],
            'risk_score': ['Low', 'Medium', 'High'],
            'state': ['Healthy', 'At_Risk', 'CVD'],
            'chest_pain': ['No', 'Yes'],
            'fatigue': ['No', 'Yes'],
            'shortness_of_breath': ['No', 'Yes'],
            'treatment': ['None', 'Medication', 'Surgery'],
            'utility': ['Low', 'Medium', 'High']
        }
        
        return self.model
    
    def discretize_variables(self, df, discretization_rules=None):
        """
        Discretize continuous variables for BN implementation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with continuous variables
        discretization_rules : dict, optional
            Custom discretization rules for variables
            
        Returns:
        --------
        df_discrete : pd.DataFrame
            Discretized dataset
        discretization_info : dict
            Information about discretization thresholds
        """
        df_discrete = df.copy()
        discretization_info = {}
        
        # Default discretization rules
        default_rules = {
            'age': {'thresholds': [40, 65], 'labels': ['Young', 'Middle', 'Old']},
            'cholesterol': {'thresholds': [200], 'labels': ['Normal', 'High']},
            'blood_pressure': {'thresholds': [140], 'labels': ['Normal', 'High']},
            'risk_score': {'thresholds': [2, 4], 'labels': ['Low', 'Medium', 'High']},
            'utility': {'thresholds': [3, 7], 'labels': ['Low', 'Medium', 'High']}
        }
        
        rules = discretization_rules or default_rules
        
        for variable, rule in rules.items():
            if variable in df_discrete.columns:
                if df_discrete[variable].dtype in ['float64', 'int64']:
                    df_discrete[variable] = pd.cut(
                        df_discrete[variable],
                        bins=[-np.inf] + rule['thresholds'] + [np.inf],
                        labels=rule['labels'],
                        include_lowest=True
                    )
                    discretization_info[variable] = rule
        
        # Update variable states based on actual data
        for col in df_discrete.columns:
            if col in self.variable_states:
                unique_values = df_discrete[col].dropna().unique()
                self.variable_states[col] = sorted(unique_values.astype(str))
        
        return df_discrete, discretization_info
    
    def estimate_cpds(self, df_discrete, method='mle', pseudo_counts=1):
        """
        Estimate Conditional Probability Distributions from data.
        
        Parameters:
        -----------
        df_discrete : pd.DataFrame
            Discretized dataset
        method : str, default='mle'
            Estimation method ('mle' or 'bayes')
        pseudo_counts : int, default=1
            Pseudo counts for Bayesian estimation
            
        Returns:
        --------
        cpds : list
            List of estimated CPDs
        """
        if self.model is None:
            raise ValueError("Must create network structure first")
        
        # Filter dataframe to only include variables in the network
        network_variables = list(self.model.nodes())
        available_variables = [var for var in network_variables if var in df_discrete.columns]
        df_filtered = df_discrete[available_variables].dropna()
        
        if method == 'mle':
            estimator = MaximumLikelihoodEstimator(self.model, df_filtered)
        else:
            estimator = BayesianEstimator(self.model, df_filtered)
        
        cpds = []
        for node in self.model.nodes():
            if node in df_filtered.columns:
                if method == 'mle':
                    cpd = estimator.estimate_cpd(node)
                else:
                    cpd = estimator.estimate_cpd(node, prior_type="dirichlet", pseudo_counts=pseudo_counts)
                cpds.append(cpd)
        
        # Add CPDs to model
        self.model.add_cpds(*cpds)
        
        # Validate model
        if self.model.check_model():
            print("Model validation successful!")
        else:
            print("Warning: Model validation failed!")
        
        return cpds
    
    def perform_exact_inference(self, evidence=None, query_variables=None):
        """
        Perform exact inference using Variable Elimination and Belief Propagation.
        
        Parameters:
        -----------
        evidence : dict, optional
            Evidence dictionary {variable: value}
        query_variables : list, optional
            Variables to query
            
        Returns:
        --------
        results : dict
            Inference results from both VE and BP
        """
        if self.model is None or not self.model.get_cpds():
            raise ValueError("Must create and parameterize model first")
        
        results = {}
        
        # Variable Elimination
        ve_inference = VariableElimination(self.model)
        
        # Belief Propagation
        bp_inference = BeliefPropagation(self.model)
        
        if query_variables is None:
            query_variables = ['state', 'treatment', 'utility']
        
        for var in query_variables:
            if var in self.model.nodes():
                # VE inference
                ve_result = ve_inference.query(variables=[var], evidence=evidence)
                
                # BP inference
                bp_result = bp_inference.query(variables=[var], evidence=evidence)
                
                results[var] = {
                    'variable_elimination': ve_result,
                    'belief_propagation': bp_result,
                    'evidence': evidence
                }
        
        return results
    
    def perform_approximate_inference(self, evidence=None, query_variables=None, 
                                    method='likelihood_weighted', n_samples=10000):
        """
        Perform approximate inference using sampling methods.
        
        Parameters:
        -----------
        evidence : dict, optional
            Evidence dictionary {variable: value}
        query_variables : list, optional
            Variables to query
        method : str, default='likelihood_weighted'
            Sampling method ('forward', 'rejection', 'likelihood_weighted')
        n_samples : int, default=10000
            Number of samples to generate
            
        Returns:
        --------
        results : dict
            Approximate inference results
        """
        if self.model is None or not self.model.get_cpds():
            raise ValueError("Must create and parameterize model first")
        
        sampler = BayesianModelSampling(self.model)
        results = {}
        
        if query_variables is None:
            query_variables = ['state', 'treatment', 'utility']
        
        if method == 'forward':
            samples = sampler.forward_sample(size=n_samples, return_type='dataframe')
        elif method == 'rejection':
            samples = sampler.rejection_sample(evidence=evidence, size=n_samples, return_type='dataframe')
        elif method == 'likelihood_weighted':
            samples = sampler.likelihood_weighted_sample(evidence=evidence, size=n_samples, return_type='dataframe')
        
        # Calculate approximate posteriors
        for var in query_variables:
            if var in samples.columns:
                if evidence and method != 'forward':
                    # For methods that handle evidence
                    posterior = samples[var].value_counts(normalize=True).sort_index()
                else:
                    # For forward sampling, filter by evidence manually
                    filtered_samples = samples.copy()
                    if evidence:
                        for ev_var, ev_val in evidence.items():
                            if ev_var in filtered_samples.columns:
                                filtered_samples = filtered_samples[filtered_samples[ev_var] == ev_val]
                    
                    if len(filtered_samples) > 0:
                        posterior = filtered_samples[var].value_counts(normalize=True).sort_index()
                    else:
                        posterior = pd.Series(dtype=float)
                
                results[var] = {
                    'posterior': posterior,
                    'method': method,
                    'n_samples': n_samples,
                    'evidence': evidence
                }
        
        return results
    
    def compare_inference_methods(self, evidence=None, query_variables=None, n_samples=10000):
        """
        Compare exact vs approximate inference results.
        
        Parameters:
        -----------
        evidence : dict, optional
            Evidence dictionary
        query_variables : list, optional
            Variables to query
        n_samples : int, default=10000
            Number of samples for approximate inference
            
        Returns:
        --------
        comparison : dict
            Comparison results with distance metrics
        """
        # Exact inference
        exact_results = self.perform_exact_inference(evidence, query_variables)
        
        # Approximate inference
        approx_results = self.perform_approximate_inference(
            evidence, query_variables, method='likelihood_weighted', n_samples=n_samples
        )
        
        comparison = {}
        
        for var in exact_results.keys():
            exact_probs = exact_results[var]['variable_elimination'].values
            approx_probs = approx_results[var]['posterior'].values
            
            # Calculate distance metrics
            l1_distance = np.sum(np.abs(exact_probs - approx_probs))
            l2_distance = np.sqrt(np.sum((exact_probs - approx_probs) ** 2))
            
            comparison[var] = {
                'exact_probs': exact_probs,
                'approx_probs': approx_probs,
                'l1_distance': l1_distance,
                'l2_distance': l2_distance,
                'max_diff': np.max(np.abs(exact_probs - approx_probs))
            }
        
        return comparison
    
    def visualize_network(self, save_path=None, figsize=(12, 8)):
        """
        Visualize the Bayesian Network structure.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if self.model is None:
            raise ValueError("Must create network structure first")
        
        plt.figure(figsize=figsize)
        
        # Create networkx graph
        G = nx.DiGraph()
        G.add_edges_from(self.model.edges())
        
        # Define node positions for better layout
        pos = self._get_node_positions()
        
        # Draw network
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=10, font_weight='bold',
                edge_color='gray', arrows=True, arrowsize=20,
                arrowstyle='->')
        
        plt.title('Cardiovascular Disease Bayesian Network', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _get_node_positions(self):
        """
        Get optimized node positions for network visualization.
        
        Returns:
        --------
        pos : dict
            Node positions for visualization
        """
        # Define positions based on medical hierarchy
        positions = {
            # Root causes (demographics)
            'age': (0, 2),
            'gender': (1, 2),
            
            # Risk factors
            'cholesterol': (0, 1),
            'blood_pressure': (1, 1),
            
            # Intermediate
            'risk_score': (0.5, 0),
            
            # State
            'state': (0.5, -1),
            
            # Symptoms
            'chest_pain': (-1, -2),
            'fatigue': (0, -2),
            'shortness_of_breath': (1, -2),
            
            # Treatment and outcome
            'treatment': (2, -1),
            'utility': (2, -2)
        }
        
        # Use spring layout for any missing nodes
        if self.model:
            existing_nodes = set(self.model.nodes())
            positioned_nodes = set(positions.keys())
            missing_nodes = existing_nodes - positioned_nodes
            
            if missing_nodes:
                spring_pos = nx.spring_layout(self.model.subgraph(missing_nodes))
                positions.update(spring_pos)
        
        return positions
    
    def analyze_independencies(self):
        """
        Analyze conditional independence relationships in the network.
        
        Returns:
        --------
        independencies : list
            List of conditional independence statements
        """
        if self.model is None:
            raise ValueError("Must create network structure first")
        
        independencies = list(self.model.get_independencies())
        
        # Categorize independencies by type
        analysis = {
            'total_independencies': len(independencies),
            'marginal_independencies': [],
            'conditional_independencies': [],
            'clinical_interpretation': []
        }
        
        for independence in independencies:
            if len(independence.event2) == 0:  # Marginal independence
                analysis['marginal_independencies'].append(independence)
            else:  # Conditional independence
                analysis['conditional_independencies'].append(independence)
        
        # Add clinical interpretations
        for independence in independencies[:5]:  # First 5 for brevity
            interpretation = self._interpret_independence_clinically(independence)
            analysis['clinical_interpretation'].append(interpretation)
        
        return analysis
    
    def _interpret_independence_clinically(self, independence):
        """
        Provide clinical interpretation of independence relationship.
        
        Parameters:
        -----------
        independence : Independence
            Independence relationship to interpret
            
        Returns:
        --------
        interpretation : str
            Clinical interpretation
        """
        var1 = list(independence.event1)[0] if independence.event1 else "Unknown"
        var2 = list(independence.event2)[0] if independence.event2 else "Unknown"
        given = list(independence.event3) if independence.event3 else []
        
        if given:
            interpretation = f"{var1} is independent of {var2} given {', '.join(given)}"
        else:
            interpretation = f"{var1} is marginally independent of {var2}"
        
        # Add clinical context
        clinical_mappings = {
            ('age', 'treatment'): "Age doesn't directly influence treatment choice (mediated by disease state)",
            ('symptoms', 'demographics'): "Symptoms depend on disease state, not directly on demographics",
            ('treatment', 'symptoms'): "Treatment effects are mediated through disease state changes"
        }
        
        for (v1, v2), clinical_meaning in clinical_mappings.items():
            if (v1 in var1.lower() and v2 in var2.lower()) or (v2 in var1.lower() and v1 in var2.lower()):
                interpretation += f" - {clinical_meaning}"
                break
        
        return interpretation


class CVDStructureLearning:
    """
    Structure learning utilities for Bayesian Networks.
    
    Implements constraint-based and score-based structure learning algorithms.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize structure learning.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducible results
        """
        self.random_state = random_state
        
    def learn_pc_structure(self, df, significance_level=0.05):
        """
        Learn network structure using PC algorithm (constraint-based).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Discretized dataset
        significance_level : float, default=0.05
            Significance level for independence tests
            
        Returns:
        --------
        learned_model : BayesianNetwork
            Learned network structure
        """
        pc = PC(df)
        learned_model = pc.estimate(significance_level=significance_level)
        
        return learned_model
    
    def learn_hc_structure(self, df, scoring_method='bic', max_indegree=3):
        """
        Learn network structure using Hill Climbing (score-based).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Discretized dataset
        scoring_method : str, default='bic'
            Scoring method ('bic', 'aic', 'k2')
        max_indegree : int, default=3
            Maximum number of parents per node
            
        Returns:
        --------
        learned_model : BayesianNetwork
            Learned network structure
        """
        if scoring_method == 'bic':
            scoring_func = BicScore(df)
        else:
            raise ValueError(f"Scoring method {scoring_method} not implemented")
        
        hc = HillClimbSearch(df)
        learned_model = hc.estimate(scoring_method=scoring_func, max_indegree=max_indegree)
        
        return learned_model
    
    def compare_structures(self, true_model, learned_model):
        """
        Compare learned structure with true/hand-designed structure.
        
        Parameters:
        -----------
        true_model : BayesianNetwork
            True or hand-designed network
        learned_model : BayesianNetwork
            Learned network structure
            
        Returns:
        --------
        comparison : dict
            Structure comparison metrics
        """
        true_edges = set(true_model.edges())
        learned_edges = set(learned_model.edges())
        
        # Calculate metrics
        tp = len(true_edges.intersection(learned_edges))  # True positives
        fp = len(learned_edges - true_edges)  # False positives
        fn = len(true_edges - learned_edges)  # False negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Structural Hamming Distance
        shd = fp + fn
        
        comparison = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'structural_hamming_distance': shd,
            'true_edges': list(true_edges),
            'learned_edges': list(learned_edges),
            'missing_edges': list(true_edges - learned_edges),
            'extra_edges': list(learned_edges - true_edges)
        }
        
        return comparison