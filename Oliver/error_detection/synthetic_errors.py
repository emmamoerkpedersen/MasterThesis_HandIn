"""
Module for generating synthetic errors in time series data.
Contains functions for injecting various types of errors (spikes, gaps, flatlines, etc.)
into clean data segments for testing error detection algorithms.
"""

import pandas as pd
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ErrorPeriod:
    """Class to store information about injected errors."""
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    error_type: str
    original_values: np.ndarray
    modified_values: np.ndarray
    parameters: Dict

class SyntheticErrorGenerator:
    """Generate synthetic errors in time series data."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize error generator with configuration.
        
        Args:
            config: Dictionary of error parameters (if None, uses default from config.py)
        """
        from error_detection.config import SYNTHETIC_ERROR_PARAMS
        self.config = config or SYNTHETIC_ERROR_PARAMS
        self.error_periods: List[ErrorPeriod] = []
    
    def find_valid_period(self, data: pd.DataFrame, 
                         duration: pd.Timedelta,
                         buffer_hours: int = 24) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Find a period where no errors have been injected (including buffer).
        
        Args:
            data: Input DataFrame
            duration: Required duration for new error
            buffer_hours: Minimum hours between errors
        
        Returns:
            start_time, end_time for new error
        """
        buffer = pd.Timedelta(hours=buffer_hours)
        
        # Get all existing error periods
        busy_periods = []
        for error in self.error_periods:
            busy_periods.append((
                error.start_time - buffer,
                error.end_time + buffer
            ))
        
        # Find gaps between busy periods
        available_periods = self._find_available_periods(data.index, busy_periods)
        
        # Select random period of sufficient duration
        valid_periods = [p for p in available_periods if p[1]-p[0] >= duration]
        if not valid_periods:
            raise ValueError("No valid periods available for error injection")
        
        chosen_period = random.choice(valid_periods)
        start_time = chosen_period[0] + buffer
        end_time = start_time + duration
        
        return start_time, end_time
    
    def inject_spike_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inject spike errors into the data."""
        modified_data = data.copy()
        config = self.config['spike']
        
        n_spikes = int(len(data) * config['frequency'])
        for _ in range(n_spikes):
            # Find valid location for spike
            start_time, end_time = self.find_valid_period(
                data, 
                pd.Timedelta(hours=1)
            )
            
            # Calculate spike magnitude
            local_std = data.loc[start_time:end_time, 'Value'].std()
            magnitude = random.uniform(*config['magnitude_range']) * local_std
            
            # Store original and create spike
            original_values = modified_data.loc[start_time:end_time, 'Value'].copy()
            modified_data.loc[start_time:end_time, 'Value'] += magnitude
            
            # Record error period
            self.error_periods.append(ErrorPeriod(
                start_time=start_time,
                end_time=end_time,
                error_type='spike',
                original_values=original_values.values,
                modified_values=modified_data.loc[start_time:end_time, 'Value'].values,
                parameters={'magnitude': magnitude}
            ))
        
        return modified_data
    
    def inject_all_errors(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject all types of errors into the data.
        
        Returns:
            modified_data, ground_truth_labels
        """
        modified_data = data.copy()
        
        # Inject errors in sequence
        modified_data = self.inject_spike_errors(modified_data)
        modified_data = self.inject_gap_errors(modified_data)
        modified_data = self.inject_flatline_errors(modified_data)
        
        # Create ground truth labels
        ground_truth = self._create_ground_truth(data.index)
        
        return modified_data, ground_truth
    
    def _create_ground_truth(self, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create DataFrame with ground truth labels."""
        ground_truth = pd.DataFrame(index=time_index)
        ground_truth['error_type'] = 'normal'
        ground_truth['original_value'] = np.nan
        ground_truth['modified_value'] = np.nan
        
        for error in self.error_periods:
            mask = (ground_truth.index >= error.start_time) & \
                  (ground_truth.index <= error.end_time)
            ground_truth.loc[mask, 'error_type'] = error.error_type
            ground_truth.loc[mask, 'original_value'] = error.original_values
            ground_truth.loc[mask, 'modified_value'] = error.modified_values
        
        return ground_truth

def inject_spike_error(data: pd.DataFrame, 
                      frequency: float = 0.01, 
                      magnitude_range: tuple = (2, 5)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inject spike errors into the time series data.
    
    Args:
        data (pd.DataFrame): Input time series data
        frequency (float): Probability of spike occurrence
        magnitude_range (tuple): Range of spike magnitudes in standard deviations
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Modified data and ground truth labels
    """
    pass

def inject_gap_error(data: pd.DataFrame,
                    frequency: float = 0.01,
                    duration_range: tuple = (1, 24)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inject gap errors (missing data) into the time series.
    
    Args:
        data (pd.DataFrame): Input time series data
        frequency (float): Probability of gap occurrence
        duration_range (tuple): Range of gap durations in hours
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Modified data and ground truth labels
    """
    pass

def inject_flatline_error(data: pd.DataFrame,
                         frequency: float = 0.01,
                         duration_range: tuple = (2, 48)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inject flatline errors into the time series.
    
    Args:
        data (pd.DataFrame): Input time series data
        frequency (float): Probability of flatline occurrence
        duration_range (tuple): Range of flatline durations in hours
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Modified data and ground truth labels
    """
    pass 