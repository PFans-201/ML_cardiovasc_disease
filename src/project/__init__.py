"""
Cardiovascular Disease Prediction & Treatment Optimization Project

This package provides utilities for:
- Data loading and preprocessing
- Exploratory data analysis and visualization  
- Feature engineering for temporal medical data
- Machine learning model development and evaluation
- Treatment optimization using reinforcement learning

Main modules:
- data_loader: Standardized data loading and preprocessing
- visualizations: Comprehensive plotting utilities for EDA
- auteda: Automated exploratory data analysis
- eda_class: Object-oriented EDA framework

Example usage:
    from project import CVDDataLoader, quick_eda_plots
    
    # Load data
    loader = CVDDataLoader()
    patients, encounters = loader.load_raw_data()
    merged = loader.merge_data()
    
    # Generate visualizations
    figures = quick_eda_plots(patients, encounters)
"""

# Import main classes and functions
from .data_loader import CVDDataLoader, load_cvd_data
from .visualizations import CVDVisualizer, quick_eda_plots
from .auteda import AutomatedEDA
# from .eda_class import *

__version__ = "0.1.0"
__author__ = "Advanced ML Group 14"

# Define public API
__all__ = [
    'CVDDataLoader',
    'load_cvd_data', 
    'CVDVisualizer',
    'quick_eda_plots',
    'AutomatedEDA'
]

