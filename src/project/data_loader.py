"""
Data Loading and Preprocessing Utilities for Cardiovascular Disease Dataset

This module provides standardized functions for loading, cleaning, and preprocessing
the longitudinal cardiovascular disease dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings
from pathlib import Path


class CVDDataLoader:
    """
    Standardized data loader for the cardiovascular disease longitudinal dataset.
    
    Handles loading, basic cleaning, and merging of patient and encounter data.
    """
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize data loader with path to raw data directory.
        
        Parameters:
        -----------
        data_path : str
            Path to directory containing patients.csv and encounters.csv
        """
        self.data_path = Path(data_path)
        self.patients = None
        self.encounters = None
        self.merged_data = None
        
    def load_raw_data(self, fix_missing: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw patient and encounter datasets.
        
        Parameters:
        -----------
        fix_missing : bool
            Whether to apply basic missing value fixes (NaN treatments -> 'None')
            
        Returns:
        --------
        patients, encounters : tuple of pd.DataFrame
            Loaded datasets
        """
        # Load datasets with semicolon separator
        patients_path = self.data_path / "patients.csv"
        encounters_path = self.data_path / "encounters.csv"
        
        if not patients_path.exists() or not encounters_path.exists():
            raise FileNotFoundError(
                f"Data files not found in {self.data_path}. "
                f"Expected: patients.csv, encounters.csv"
            )
            
        self.patients = pd.read_csv(patients_path, sep=';')
        self.encounters = pd.read_csv(encounters_path, sep=';')
        
        # Basic data validation
        self._validate_data()
        
        # Fix missing treatment values (NaN -> 'None')
        if fix_missing:
            self.encounters['treatment'] = self.encounters['treatment'].fillna('None')
            
        print(f"✅ Loaded {len(self.patients):,} patients and {len(self.encounters):,} encounters")
        print(f"📊 Average encounters per patient: {len(self.encounters) / len(self.patients):.1f}")
        
        return self.patients.copy(), self.encounters.copy()
    
    def _validate_data(self):
        """Perform basic data validation checks."""
        # Check expected columns
        expected_patient_cols = {
            'patient_id', 'age', 'sex', 'bmi', 'smoker', 
            'family_history', 'hypertension', 'risk_score', 'initial_state'
        }
        expected_encounter_cols = {
            'patient_id', 'time', 'state', 'treatment', 'chest_pain', 
            'fatigue', 'shortness_of_breath', 'systolic_bp', 'cholesterol', 
            'glucose', 'troponin', 'utility'
        }
        
        missing_patient_cols = expected_patient_cols - set(self.patients.columns)
        missing_encounter_cols = expected_encounter_cols - set(self.encounters.columns)
        
        if missing_patient_cols:
            warnings.warn(f"Missing patient columns: {missing_patient_cols}")
        if missing_encounter_cols:
            warnings.warn(f"Missing encounter columns: {missing_encounter_cols}")
            
        # Check patient ID consistency
        patient_ids_patients = set(self.patients['patient_id'])
        patient_ids_encounters = set(self.encounters['patient_id'])
        
        if patient_ids_patients != patient_ids_encounters:
            warnings.warn("Patient IDs don't match between datasets")
            
        # Check time points consistency
        encounters_per_patient = self.encounters.groupby('patient_id').size()
        if not (encounters_per_patient == 8).all():
            warnings.warn("Not all patients have exactly 8 time points")
    
    def merge_data(self, include_time_features: bool = True) -> pd.DataFrame:
        """
        Merge patient and encounter data into analysis-ready format.
        
        Parameters:
        -----------
        include_time_features : bool
            Whether to add derived time-based features
            
        Returns:
        --------
        merged_df : pd.DataFrame
            Combined dataset with patient info merged to each encounter
        """
        if self.patients is None or self.encounters is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
            
        # Merge encounters with patient data
        merged = self.encounters.merge(self.patients, on='patient_id', how='left')
        
        # Add time-based features if requested
        if include_time_features:
            merged = self._add_time_features(merged)
            
        # Sort by patient and time for consistency
        merged = merged.sort_values(['patient_id', 'time']).reset_index(drop=True)
        
        self.merged_data = merged
        print(f"✅ Merged data: {len(merged):,} rows × {len(merged.columns)} columns")
        
        return merged.copy()
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived time-based features."""
        df = df.copy()
        
        # Time indicators
        df['is_baseline'] = (df['time'] == 0).astype(int)
        df['is_final'] = (df['time'] == 7).astype(int)
        df['time_squared'] = df['time'] ** 2
        df['time_normalized'] = df['time'] / 7  # Normalize to [0,1]
        
        # Time since baseline
        df['follow_up_duration'] = df['time']
        
        return df
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive data summary statistics.
        
        Returns:
        --------
        summary : dict
            Dictionary containing data summary statistics
        """
        if self.patients is None or self.encounters is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
            
        summary = {
            'patients': {
                'total_count': len(self.patients),
                'age_stats': self.patients['age'].describe(),
                'sex_distribution': self.patients['sex'].value_counts(),
                'smoker_rate': self.patients['smoker'].mean(),
                'family_history_rate': self.patients['family_history'].mean(),
                'hypertension_rate': self.patients['hypertension'].mean(),
                'initial_state_distribution': self.patients['initial_state'].value_counts(),
                'risk_score_distribution': self.patients['risk_score'].value_counts().sort_index()
            },
            'encounters': {
                'total_count': len(self.encounters),
                'encounters_per_patient': self.encounters.groupby('patient_id').size().describe(),
                'state_distribution': self.encounters['state'].value_counts(),
                'treatment_distribution': self.encounters['treatment'].value_counts(),
                'utility_stats': self.encounters['utility'].describe(),
                'missing_data_rates': {
                    col: self.encounters[col].isnull().mean() 
                    for col in self.encounters.columns 
                    if self.encounters[col].isnull().any()
                }
            },
            'data_quality': {
                'duplicate_encounters': self.encounters.duplicated(['patient_id', 'time']).sum(),
                'patients_with_incomplete_follow_up': (
                    self.encounters.groupby('patient_id').size() != 8
                ).sum()
            }
        }
        
        return summary
    
    def get_patient_trajectories(self, patient_ids: Optional[list] = None, 
                               n_sample: int = 10) -> pd.DataFrame:
        """
        Extract patient trajectories for visualization or analysis.
        
        Parameters:
        -----------
        patient_ids : list, optional
            Specific patient IDs to extract. If None, sample randomly.
        n_sample : int
            Number of patients to sample if patient_ids not provided
            
        Returns:
        --------
        trajectories : pd.DataFrame
            Patient trajectories with all encounters
        """
        if self.merged_data is None:
            self.merge_data()
            
        if patient_ids is None:
            # Sample random patients
            all_patients = self.merged_data['patient_id'].unique()
            patient_ids = np.random.choice(all_patients, size=min(n_sample, len(all_patients)), 
                                         replace=False)
        
        trajectories = self.merged_data[
            self.merged_data['patient_id'].isin(patient_ids)
        ].copy()
        
        return trajectories.sort_values(['patient_id', 'time'])


def load_cvd_data(data_path: str = "data/raw", 
                  merge: bool = True, 
                  add_time_features: bool = True) -> Tuple[pd.DataFrame, ...]:
    """
    Convenience function for quick data loading.
    
    Parameters:
    -----------
    data_path : str
        Path to raw data directory
    merge : bool
        Whether to return merged data or separate datasets
    add_time_features : bool
        Whether to add time-based derived features
        
    Returns:
    --------
    data : pd.DataFrame or tuple
        If merge=True: merged dataset
        If merge=False: (patients, encounters) tuple
    """
    loader = CVDDataLoader(data_path)
    patients, encounters = loader.load_raw_data()
    
    if merge:
        merged = loader.merge_data(include_time_features=add_time_features)
        return merged
    else:
        return patients, encounters


# Example usage and testing
if __name__ == "__main__":
    # Test the data loader
    try:
        loader = CVDDataLoader()
        patients, encounters = loader.load_raw_data()
        merged = loader.merge_data()
        
        print("\n📊 Data Summary:")
        summary = loader.get_data_summary()
        
        print(f"Patients: {summary['patients']['total_count']:,}")
        print(f"Encounters: {summary['encounters']['total_count']:,}")
        print(f"Missing data rates: {summary['encounters']['missing_data_rates']}")
        
        print("\n✅ Data loader working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing data loader: {e}")