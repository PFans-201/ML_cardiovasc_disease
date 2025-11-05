"""
Gaussian Mixture Model utilities for cardiovascular disease analysis.

This module provides utilities for applying GMM to patient data for clustering
and risk stratification as required in Milestone M1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class CVDGaussianMixture:
    """
    Gaussian Mixture Model for cardiovascular disease patient clustering.
    
    Provides comprehensive GMM analysis including:
    - Optimal component number selection
    - Patient risk stratification
    - Cluster interpretation and visualization
    - Clinical insights extraction
    """
    
    def __init__(self, random_state=42):
        """
        Initialize CVD Gaussian Mixture Model.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducible results
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.gmm = None
        self.n_components_optimal = None
        self.cluster_labels = None
        self.feature_names = None
        
    def select_optimal_components(self, X, max_components=10, cv_folds=5):
        """
        Select optimal number of components using cross-validation and information criteria.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix (should be imputed and standardized)
        max_components : int, default=10
            Maximum number of components to test
        cv_folds : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        results_df : pd.DataFrame
            Results with BIC, AIC, silhouette scores for each n_components
        """
        n_range = range(2, max_components + 1)
        results = []
        
        for n in n_range:
            # Fit GMM
            gmm = GaussianMixture(
                n_components=n,
                random_state=self.random_state,
                covariance_type='full'
            )
            gmm.fit(X)
            
            # Calculate metrics
            labels = gmm.predict(X)
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
            calinski = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else -1
            
            results.append({
                'n_components': n,
                'BIC': bic,
                'AIC': aic,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski
            })
        
        results_df = pd.DataFrame(results)
        
        # Select optimal based on BIC (lower is better)
        self.n_components_optimal = results_df.loc[results_df['BIC'].idxmin(), 'n_components']
        
        return results_df
    
    def fit_optimal_gmm(self, X, feature_names=None):
        """
        Fit GMM with optimal number of components.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        feature_names : list, optional
            Names of features for interpretation
            
        Returns:
        --------
        self : CVDGaussianMixture
            Fitted model
        """
        if self.n_components_optimal is None:
            raise ValueError("Must run select_optimal_components first")
            
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Fit final GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components_optimal,
            random_state=self.random_state,
            covariance_type='full'
        )
        self.gmm.fit(X)
        self.cluster_labels = self.gmm.predict(X)
        
        return self
    
    def analyze_clusters(self, X, df_original=None):
        """
        Analyze and interpret clusters.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix used for clustering
        df_original : pd.DataFrame, optional
            Original dataframe with patient information
            
        Returns:
        --------
        cluster_summary : pd.DataFrame
            Summary statistics for each cluster
        """
        if self.gmm is None:
            raise ValueError("Must fit GMM first")
            
        # Create cluster summary
        cluster_data = pd.DataFrame(X, columns=self.feature_names)
        cluster_data['cluster'] = self.cluster_labels
        
        # Calculate cluster statistics
        cluster_summary = cluster_data.groupby('cluster').agg({
            col: ['mean', 'std', 'count'] for col in self.feature_names
        }).round(3)
        
        # Add cluster sizes and proportions
        cluster_sizes = pd.Series(self.cluster_labels).value_counts().sort_index()
        cluster_proportions = cluster_sizes / len(self.cluster_labels)
        
        # If original dataframe provided, add clinical interpretations
        if df_original is not None:
            df_with_clusters = df_original.copy()
            df_with_clusters['cluster'] = self.cluster_labels
            
            # Analyze risk scores and states by cluster
            if 'risk_score' in df_original.columns:
                risk_by_cluster = df_with_clusters.groupby('cluster')['risk_score'].agg(['mean', 'std'])
                
            if 'state' in df_original.columns:
                state_by_cluster = df_with_clusters.groupby(['cluster', 'state']).size().unstack(fill_value=0)
                state_proportions = state_by_cluster.div(state_by_cluster.sum(axis=1), axis=0)
        
        return cluster_summary
    
    def visualize_clusters(self, X, save_path=None, figsize=(15, 10)):
        """
        Visualize clusters using various plots.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        save_path : str, optional
            Path to save the figure
        figsize : tuple, default=(15, 10)
            Figure size
        """
        if self.gmm is None:
            raise ValueError("Must fit GMM first")
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Cluster distribution
        axes[0, 0].hist(self.cluster_labels, bins=self.n_components_optimal, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Patients')
        
        # 2. First two principal components
        if X.shape[1] >= 2:
            scatter = axes[0, 1].scatter(X[:, 0], X[:, 1], c=self.cluster_labels, 
                                       cmap='viridis', alpha=0.6)
            axes[0, 1].set_title('Clusters in Feature Space (First 2 Components)')
            axes[0, 1].set_xlabel(self.feature_names[0] if self.feature_names else 'Feature 0')
            axes[0, 1].set_ylabel(self.feature_names[1] if self.feature_names else 'Feature 1')
            plt.colorbar(scatter, ax=axes[0, 1])
        
        # 3. Cluster means heatmap
        cluster_data = pd.DataFrame(X, columns=self.feature_names)
        cluster_data['cluster'] = self.cluster_labels
        cluster_means = cluster_data.groupby('cluster')[self.feature_names].mean()
        
        sns.heatmap(cluster_means.T, annot=True, cmap='RdYlBu_r', ax=axes[0, 2])
        axes[0, 2].set_title('Cluster Mean Values')
        axes[0, 2].set_xlabel('Cluster')
        
        # 4. Log-likelihood by component
        if hasattr(self.gmm, 'lower_bound_'):
            axes[1, 0].plot(self.gmm.lower_bound_)
            axes[1, 0].set_title('GMM Convergence')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Log-likelihood')
        
        # 5. Component weights
        axes[1, 1].bar(range(self.n_components_optimal), self.gmm.weights_)
        axes[1, 1].set_title('Component Weights')
        axes[1, 1].set_xlabel('Component')
        axes[1, 1].set_ylabel('Weight')
        
        # 6. Feature importance (based on component means)
        feature_importance = np.std(cluster_means.values, axis=0)
        axes[1, 2].barh(range(len(self.feature_names)), feature_importance)
        axes[1, 2].set_yticks(range(len(self.feature_names)))
        axes[1, 2].set_yticklabels(self.feature_names, rotation=0)
        axes[1, 2].set_title('Feature Importance for Clustering')
        axes[1, 2].set_xlabel('Standard Deviation Across Clusters')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def predict_risk_levels(self, X):
        """
        Assign risk levels to clusters based on average risk scores.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
            
        Returns:
        --------
        risk_mapping : dict
            Mapping from cluster to risk level
        """
        if self.gmm is None:
            raise ValueError("Must fit GMM first")
            
        # Calculate cluster centers' average values
        cluster_data = pd.DataFrame(X, columns=self.feature_names)
        cluster_data['cluster'] = self.cluster_labels
        cluster_means = cluster_data.groupby('cluster')[self.feature_names].mean()
        
        # Simple risk scoring based on feature means
        # Higher values in most features indicate higher risk
        risk_scores = cluster_means.mean(axis=1)
        
        # Assign risk levels
        risk_levels = ['Low', 'Medium', 'High']
        risk_quantiles = np.quantile(risk_scores, [0.33, 0.67])
        
        risk_mapping = {}
        for cluster_id, score in risk_scores.items():
            if score <= risk_quantiles[0]:
                risk_mapping[cluster_id] = 'Low'
            elif score <= risk_quantiles[1]:
                risk_mapping[cluster_id] = 'Medium'
            else:
                risk_mapping[cluster_id] = 'High'
                
        return risk_mapping
    
    def generate_insights(self, X, df_original=None):
        """
        Generate clinical insights from clustering results.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        df_original : pd.DataFrame, optional
            Original dataframe with clinical information
            
        Returns:
        --------
        insights : dict
            Dictionary containing clinical insights
        """
        if self.gmm is None:
            raise ValueError("Must fit GMM first")
            
        insights = {}
        
        # Basic cluster information
        insights['n_clusters'] = self.n_components_optimal
        insights['cluster_sizes'] = pd.Series(self.cluster_labels).value_counts().sort_index().to_dict()
        
        # Risk level mapping
        risk_mapping = self.predict_risk_levels(X)
        insights['risk_levels'] = risk_mapping
        
        # Feature analysis
        cluster_data = pd.DataFrame(X, columns=self.feature_names)
        cluster_data['cluster'] = self.cluster_labels
        cluster_means = cluster_data.groupby('cluster')[self.feature_names].mean()
        
        insights['cluster_characteristics'] = {}
        for cluster_id in range(self.n_components_optimal):
            characteristics = []
            cluster_mean = cluster_means.loc[cluster_id]
            
            # Identify distinctive features (above/below overall mean)
            overall_mean = pd.DataFrame(X, columns=self.feature_names).mean()
            
            for feature in self.feature_names:
                diff = cluster_mean[feature] - overall_mean[feature]
                if abs(diff) > 0.5 * overall_mean.std():  # Significant difference
                    if diff > 0:
                        characteristics.append(f"High {feature}")
                    else:
                        characteristics.append(f"Low {feature}")
            
            insights['cluster_characteristics'][cluster_id] = {
                'risk_level': risk_mapping[cluster_id],
                'size': insights['cluster_sizes'][cluster_id],
                'distinctive_features': characteristics
            }
        
        return insights


def plot_model_selection_results(results_df, save_path=None):
    """
    Plot model selection results for different numbers of components.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from select_optimal_components
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # BIC and AIC
    axes[0, 0].plot(results_df['n_components'], results_df['BIC'], 'o-', label='BIC')
    axes[0, 0].plot(results_df['n_components'], results_df['AIC'], 's-', label='AIC')
    axes[0, 0].set_xlabel('Number of Components')
    axes[0, 0].set_ylabel('Information Criterion')
    axes[0, 0].set_title('Model Selection: Information Criteria')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Silhouette Score
    axes[0, 1].plot(results_df['n_components'], results_df['silhouette_score'], 'o-', color='green')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Calinski-Harabasz Score
    axes[1, 0].plot(results_df['n_components'], results_df['calinski_harabasz_score'], 'o-', color='red')
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title('Calinski-Harabasz Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table_data = results_df.round(3)
    table = axes[1, 1].table(cellText=table_data.values, colLabels=table_data.columns,
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    axes[1, 1].set_title('Detailed Results')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()