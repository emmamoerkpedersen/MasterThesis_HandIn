"""Statistical models for anomaly detection."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Union
from scipy import stats

class StatisticalDetector:
    def __init__(self, window_size=96, z_threshold=3.0):
        """
        Initialize statistical detector.
        
        Args:
            window_size: Size of rolling window (96 points = 24 hours with 15min data)
            z_threshold: Z-score threshold for anomaly detection
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.global_mean = None
        self.global_std = None
    
    def fit(self, data: pd.DataFrame):
        """Learn statistical properties of the data."""
        values = data['Value'].values
        self.global_mean = np.mean(values)
        self.global_std = np.std(values)
        print(f"Learned global statistics: mean={self.global_mean:.2f}, std={self.global_std:.2f}")
        return self
    
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods.
        
        Uses:
        1. Z-score from global distribution
        2. Rolling mean and std for local context
        3. Rate of change (derivative)
        """
        # Get values
        values = data['Value'].values
        
        # 1. Global Z-score
        global_z = np.abs((values - self.global_mean) / self.global_std)
        
        # 2. Rolling statistics
        df = pd.DataFrame({'value': values})
        rolling_mean = df['value'].rolling(window=self.window_size, center=True).mean()
        rolling_std = df['value'].rolling(window=self.window_size, center=True).std()
        
        # Handle NaN values at edges
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
        rolling_std = rolling_std.fillna(self.global_std)
        
        # Local Z-score
        local_z = np.abs((values - rolling_mean) / rolling_std)
        
        # 3. Rate of change
        diff = np.diff(values, prepend=values[0])
        abs_diff = np.abs(diff)
        
        # Get rolling stats for rate of change
        rolling_diff_mean = pd.Series(abs_diff).rolling(window=self.window_size, center=True).mean()
        rolling_diff_std = pd.Series(abs_diff).rolling(window=self.window_size, center=True).std()
        
        # Handle NaN values
        rolling_diff_mean = rolling_diff_mean.fillna(method='bfill').fillna(method='ffill')
        rolling_diff_std = rolling_diff_std.fillna(np.std(abs_diff))
        
        # Z-score for rate of change
        diff_z = np.abs((abs_diff - rolling_diff_mean) / rolling_diff_std)
        
        # Combine scores (weighted average)
        combined_score = (0.4 * global_z + 0.4 * local_z + 0.2 * diff_z)
        
        # Normalize to [0,1]
        min_score = np.min(combined_score)
        max_score = np.max(combined_score)
        normalized_score = (combined_score - min_score) / (max_score - min_score + 1e-10)
        
        # Detect anomalies
        anomalies = combined_score > self.z_threshold
        
        # Create reconstructed values (rolling mean)
        reconstructed = rolling_mean.values
        
        return {
            'anomaly_mask': anomalies,
            'confidence_scores': normalized_score,
            'reconstructed': reconstructed,
            'global_z': global_z,
            'local_z': local_z,
            'diff_z': diff_z
        } 