"""
Visualization utilities for cardiovascular disease dataset analysis.

This module provides standardized plotting functions for exploratory data analysis,
model evaluation, and result presentation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import warnings
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Set style defaults
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CVDVisualizer:
    """
    Comprehensive visualization toolkit for cardiovascular disease dataset.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer with default figure settings.
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size (width, height)
        """
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
        
    def plot_patient_demographics(self, patients_df: pd.DataFrame, 
                                 figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Create comprehensive patient demographics overview.
        
        Parameters:
        -----------
        patients_df : pd.DataFrame
            Patient demographics dataset
        figsize : tuple, optional
            Figure size override
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Complete demographics figure
        """
        if figsize is None:
            figsize = (16, 12)
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Patient Demographics Overview', fontsize=16, fontweight='bold')
        
        # Age distribution
        axes[0, 0].hist(patients_df['age'], bins=30, alpha=0.7, color=self.colors[0])
        axes[0, 0].axvline(patients_df['age'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {patients_df["age"].mean():.1f}')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age (years)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        
        # Sex distribution
        sex_counts = patients_df['sex'].value_counts()
        axes[0, 1].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%',
                      colors=self.colors[:2])
        axes[0, 1].set_title('Sex Distribution')
        
        # BMI distribution
        axes[0, 2].hist(patients_df['bmi'], bins=30, alpha=0.7, color=self.colors[1])
        axes[0, 2].axvline(patients_df['bmi'].mean(), color='red', linestyle='--',
                          label=f'Mean: {patients_df["bmi"].mean():.1f}')
        axes[0, 2].set_title('BMI Distribution')
        axes[0, 2].set_xlabel('BMI')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend()
        
        # Risk factors
        risk_factors = ['smoker', 'family_history', 'hypertension']
        risk_rates = [patients_df[col].mean() for col in risk_factors]
        risk_labels = ['Smoker', 'Family History', 'Hypertension']
        
        bars = axes[1, 0].bar(risk_labels, risk_rates, color=self.colors[2:5])
        axes[1, 0].set_title('Risk Factor Prevalence')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, risk_rates):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{rate:.2f}', ha='center', va='bottom')
        
        # Risk score distribution
        risk_counts = patients_df['risk_score'].value_counts().sort_index()
        axes[1, 1].bar(risk_counts.index, risk_counts.values, color=self.colors[5])
        axes[1, 1].set_title('Risk Score Distribution')
        axes[1, 1].set_xlabel('Risk Score')
        axes[1, 1].set_ylabel('Count')
        
        # Initial state distribution
        state_counts = patients_df['initial_state'].value_counts()
        axes[1, 2].pie(state_counts.values, labels=state_counts.index, 
                      autopct='%1.1f%%', colors=self.colors[6:9])
        axes[1, 2].set_title('Initial Disease State')
        
        plt.tight_layout()
        return fig
    
    def plot_disease_progression(self, encounters_df: pd.DataFrame,
                               n_patients: int = 20,
                               figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Visualize disease state progression over time for sample patients.
        
        Parameters:
        -----------
        encounters_df : pd.DataFrame
            Encounter data with patient trajectories
        n_patients : int
            Number of patients to display
        figsize : tuple, optional
            Figure size override
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Disease progression figure
        """
        if figsize is None:
            figsize = (14, 10)
            
        # Sample patients for visualization
        patient_ids = encounters_df['patient_id'].unique()[:n_patients]
        sample_data = encounters_df[encounters_df['patient_id'].isin(patient_ids)]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle('Disease Progression Analysis', fontsize=16, fontweight='bold')
        
        # Individual patient trajectories
        state_mapping = {'Healthy': 0, 'Early': 1, 'Advanced': 2}
        colors_states = {'Healthy': 'green', 'Early': 'orange', 'Advanced': 'red'}
        
        for patient_id in patient_ids:
            patient_data = sample_data[sample_data['patient_id'] == patient_id]
            patient_data = patient_data.sort_values('time')
            
            states_numeric = [state_mapping[state] for state in patient_data['state']]
            ax1.plot(patient_data['time'], states_numeric, 'o-', alpha=0.6, linewidth=1)
        
        ax1.set_title(f'Individual Disease Trajectories (n={n_patients} patients)')
        ax1.set_xlabel('Time Point')
        ax1.set_ylabel('Disease State')
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['Healthy', 'Early', 'Advanced'])
        ax1.grid(True, alpha=0.3)
        
        # Population-level state distribution over time
        state_by_time = encounters_df.groupby(['time', 'state']).size().unstack(fill_value=0)
        state_by_time_prop = state_by_time.div(state_by_time.sum(axis=1), axis=0)
        
        bottom = np.zeros(len(state_by_time_prop))
        for state in ['Healthy', 'Early', 'Advanced']:
            if state in state_by_time_prop.columns:
                ax2.bar(state_by_time_prop.index, state_by_time_prop[state], 
                       bottom=bottom, label=state, color=colors_states[state], alpha=0.8)
                bottom += state_by_time_prop[state]
        
        ax2.set_title('Population Disease State Distribution Over Time')
        ax2.set_xlabel('Time Point')
        ax2.set_ylabel('Proportion of Patients')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_treatment_analysis(self, encounters_df: pd.DataFrame,
                              figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Analyze treatment patterns and effectiveness.
        
        Parameters:
        -----------
        encounters_df : pd.DataFrame
            Encounter data with treatments and outcomes
        figsize : tuple, optional
            Figure size override
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Treatment analysis figure
        """
        if figsize is None:
            figsize = (16, 10)
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Treatment Analysis', fontsize=16, fontweight='bold')
        
        # Treatment distribution
        treatment_counts = encounters_df['treatment'].value_counts()
        axes[0, 0].pie(treatment_counts.values, labels=treatment_counts.index,
                      autopct='%1.1f%%', colors=self.colors)
        axes[0, 0].set_title('Overall Treatment Distribution')
        
        # Treatment by disease state
        treatment_by_state = pd.crosstab(encounters_df['state'], encounters_df['treatment'])
        treatment_by_state_prop = treatment_by_state.div(treatment_by_state.sum(axis=1), axis=0)
        
        treatment_by_state_prop.plot(kind='bar', ax=axes[0, 1], stacked=True, 
                                   color=self.colors[:len(treatment_by_state_prop.columns)])
        axes[0, 1].set_title('Treatment Distribution by Disease State')
        axes[0, 1].set_xlabel('Disease State')
        axes[0, 1].set_ylabel('Proportion')
        axes[0, 1].legend(title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Utility by treatment
        utility_by_treatment = encounters_df.groupby('treatment')['utility'].agg(['mean', 'std'])
        
        bars = axes[1, 0].bar(utility_by_treatment.index, utility_by_treatment['mean'],
                             yerr=utility_by_treatment['std'], capsize=5, 
                             color=self.colors[:len(utility_by_treatment)])
        axes[1, 0].set_title('Average Utility by Treatment')
        axes[1, 0].set_xlabel('Treatment')
        axes[1, 0].set_ylabel('Utility Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, utility_by_treatment['mean']):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Treatment over time
        treatment_by_time = pd.crosstab(encounters_df['time'], encounters_df['treatment'])
        treatment_by_time.plot(kind='line', ax=axes[1, 1], marker='o',
                             color=self.colors[:len(treatment_by_time.columns)])
        axes[1, 1].set_title('Treatment Usage Over Time')
        axes[1, 1].set_xlabel('Time Point')
        axes[1, 1].set_ylabel('Number of Patients')
        axes[1, 1].legend(title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_missing_data_analysis(self, encounters_df: pd.DataFrame,
                                 figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Visualize missing data patterns.
        
        Parameters:
        -----------
        encounters_df : pd.DataFrame
            Encounter data with missing values
        figsize : tuple, optional
            Figure size override
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Missing data analysis figure
        """
        if figsize is None:
            figsize = (14, 8)
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Missing Data Analysis', fontsize=16, fontweight='bold')
        
        # Missing data rates by column
        missing_rates = encounters_df.isnull().mean().sort_values(ascending=True)
        missing_rates = missing_rates[missing_rates > 0]  # Only show columns with missing data
        
        if len(missing_rates) > 0:
            bars = ax1.barh(range(len(missing_rates)), missing_rates.values, 
                           color=self.colors[0])
            ax1.set_yticks(range(len(missing_rates)))
            ax1.set_yticklabels(missing_rates.index)
            ax1.set_xlabel('Missing Data Rate')
            ax1.set_title('Missing Data by Variable')
            
            # Add percentage labels
            for i, (bar, rate) in enumerate(zip(bars, missing_rates.values)):
                ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{rate:.1%}', ha='left', va='center', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No missing data found', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Missing Data by Variable')
        
        # Missing data heatmap (sample of patients)
        sample_size = min(100, len(encounters_df))
        sample_data = encounters_df.sample(sample_size)
        
        # Create binary missing data matrix
        missing_matrix = sample_data.isnull().astype(int)
        
        if missing_matrix.sum().sum() > 0:  # If there's any missing data
            sns.heatmap(missing_matrix.T, cbar=True, cmap='Reds', 
                       ax=ax2, xticklabels=False)
            ax2.set_title(f'Missing Data Pattern (n={sample_size} encounters)')
            ax2.set_xlabel('Encounter Index')
            ax2.set_ylabel('Variables')
        else:
            ax2.text(0.5, 0.5, 'No missing data to display', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Missing Data Pattern')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                              figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Create correlation matrix for numeric variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with numeric variables
        figsize : tuple, optional
            Figure size override
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Correlation matrix figure
        """
        if figsize is None:
            figsize = (12, 10)
            
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        
        ax.set_title('Correlation Matrix (Numeric Variables)', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig


# Convenience functions
def quick_eda_plots(patients_df: pd.DataFrame, encounters_df: pd.DataFrame) -> List[Figure]:
    """
    Generate a complete set of EDA plots quickly.
    
    Parameters:
    -----------
    patients_df : pd.DataFrame
        Patient demographics data
    encounters_df : pd.DataFrame
        Encounter data
        
    Returns:
    --------
    figures : list of matplotlib.figure.Figure
        List of all generated figures
    """
    visualizer = CVDVisualizer()
    
    figures = []
    
    try:
        # Patient demographics
        fig1 = visualizer.plot_patient_demographics(patients_df)
        figures.append(fig1)
        
        # Disease progression
        fig2 = visualizer.plot_disease_progression(encounters_df)
        figures.append(fig2)
        
        # Treatment analysis
        fig3 = visualizer.plot_treatment_analysis(encounters_df)
        figures.append(fig3)
        
        # Missing data analysis
        fig4 = visualizer.plot_missing_data_analysis(encounters_df)
        figures.append(fig4)
        
        # Correlation matrix (merged data)
        merged = encounters_df.merge(patients_df, on='patient_id')
        fig5 = visualizer.plot_correlation_matrix(merged)
        figures.append(fig5)
        
        print(f"✅ Generated {len(figures)} EDA plots successfully!")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        
    return figures


if __name__ == "__main__":
    # Test visualization with dummy data
    print("Testing CVD visualizer...")
    print("Run this module after loading real data for full functionality.")