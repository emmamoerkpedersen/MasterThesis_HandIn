"""
Anomaly detection utilities for time series data.

This module contains functions for detecting anomalies in time series data,
using various statistical and machine learning approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

def calculate_anomaly_score(error: float, error_mean: float, error_std: float, local_z_score: Optional[float] = None) -> float:
    """
    Calculate a probabilistic anomaly score using both prediction error and local statistics.
    
    Args:
        error: The prediction error value
        error_mean: Mean of the error distribution
        error_std: Standard deviation of the error distribution
        local_z_score: Optional z-score from local statistical analysis
        
    Returns:
        A normalized anomaly score between 0-1, with 1 being most likely an anomaly
    """
    # Avoid division by zero
    if error_std <= 1e-10:
        error_std = 1.0
        
    # Calculate z-score for the error (how many standard deviations from mean)
    error_z_score = (error - error_mean) / error_std
    
    # Convert to probability using Gaussian properties
    # This gives us probability of seeing a value this extreme
    error_prob = 1 - np.exp(-0.5 * (error_z_score ** 2))
    
    # If we have local statistical information, factor it in
    if local_z_score is not None:
        # Similar probability conversion for local stats
        local_prob = 1 - np.exp(-0.5 * (local_z_score ** 2))
        
        # Combine probabilities (geometric mean handles uncertainties better)
        combined_score = np.sqrt(error_prob * local_prob)
        return combined_score
    
    return error_prob

def find_anomalies(
    errors: np.ndarray,
    timestamps: List, 
    original_values: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    percentile_threshold: float = 95,
    min_score_threshold: float = 0.8,
    window_size: int = 48
) -> List[Dict]:
    """
    Find anomalies in time series data using prediction errors and local statistics.
    
    Args:
        errors: Array of prediction or reconstruction errors
        timestamps: List of timestamps corresponding to each data point
        original_values: Original values from the time series
        predictions: Predicted values (optional)
        percentile_threshold: Percentile for simple error thresholding
        min_score_threshold: Minimum probabilistic score to qualify as anomaly
        window_size: Size of rolling window for local statistics

    Returns:
        List of dictionaries with anomaly information
    """
    anomalies = []
    
    # Calculate threshold and distribution statistics
    error_threshold = np.percentile(errors, percentile_threshold)
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    
    # Calculate rolling statistics for local context
    values_series = pd.Series(original_values)
    rolling_mean = values_series.rolling(window=window_size, center=True).mean()
    rolling_std = values_series.rolling(window=window_size, center=True).std()
    
    # Process each point
    for i, (ts, error) in enumerate(zip(timestamps, errors)):
        # Skip if we don't have enough context
        if i < window_size//2 or i >= len(original_values) - window_size//2:
            continue
        
        # Calculate local statistical z-score if possible
        local_z_score = None
        if i < len(rolling_mean) and i < len(rolling_std) and rolling_std[i] > 0:
            local_z_score = abs(original_values[i] - rolling_mean[i]) / rolling_std[i]
        
        # Calculate probabilistic anomaly score
        anomaly_score = calculate_anomaly_score(
            error=error,
            error_mean=error_mean,
            error_std=error_std,
            local_z_score=local_z_score
        )
        
        # Determine anomaly types
        anomaly_types = []
        if error > error_threshold:
            anomaly_types.append('prediction')
        
        if local_z_score is not None and local_z_score > 3.0:
            anomaly_types.append('statistical')
        
        # Only flag as anomaly if probabilistic score is high enough
        if anomaly_score >= min_score_threshold:
            anomaly_info = {
                'timestamp': ts,
                'index': i,
                'error': error,
                'error_threshold': error_threshold,
                'probabilistic_score': anomaly_score,
                'original_value': original_values[i],
                'local_mean': rolling_mean[i] if i < len(rolling_mean) else None,
                'local_std': rolling_std[i] if i < len(rolling_std) else None,
                'detection_methods': anomaly_types
            }
            
            # Add predicted value if available
            if predictions is not None and i < len(predictions):
                anomaly_info['predicted_value'] = predictions[i]
                
            anomalies.append(anomaly_info)
    
    # Sort anomalies by confidence score
    return sorted(anomalies, key=lambda x: x.get('probabilistic_score', 0), reverse=True) 