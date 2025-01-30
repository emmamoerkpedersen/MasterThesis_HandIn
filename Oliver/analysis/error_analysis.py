import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plotly_resampler import FigureResampler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class ErrorAnalyzer:
    def __init__(self, raw_df: pd.DataFrame, edt_df: pd.DataFrame = None, 
                 rain_df: pd.DataFrame = None,  # Add rain_df parameter
                 value_column: str = 'Value', timestamp_column: str = 'Date'):
        """
        Initialize the analyzer with raw, edited, and rainfall dataframes.
        
        Args:
            raw_df: DataFrame with raw water level measurements
            edt_df: DataFrame with edited water level measurements (optional)
            rain_df: DataFrame with rainfall data (optional)
            value_column: Name of the column containing water level values
            timestamp_column: Name of the column containing timestamps
        """
        self.df = raw_df.copy()
        self.edt_df = edt_df.copy() if edt_df is not None else None
        self.rain_df = rain_df.copy() if rain_df is not None else None
        self.value_column = value_column
        self.timestamp_column = timestamp_column
        
        # Process rainfall data if available
        if rain_df is not None:
            # Resample rainfall to daily sums
            self.rain_df = (rain_df
                          .set_index('datetime')
                          .resample('D')['precipitation (mm)']
                          .sum()
                          .reset_index())
        else:
            self.rain_df = None
        
        # Sort by timestamp and reset index
        self.df = self.df.sort_values(timestamp_column).reset_index(drop=True)
        if self.edt_df is not None:
            self.edt_df = self.edt_df.sort_values(timestamp_column).reset_index(drop=True)
        
        # Calculate time differences and rolling statistics
        self.df['time_diff'] = self.df[timestamp_column].diff().dt.total_seconds() / 3600
        self.df['rolling_mean'] = self.df[value_column].rolling(window=296, min_periods=1).mean()
        self.df['rolling_std'] = self.df[value_column].rolling(window=296, min_periods=1).std()
        
        # Update sampling constants based on 25-minute intervals
        self.SAMPLES_PER_HOUR = 60/25  # 2.4 samples per hour
        self.SAMPLES_PER_DAY = self.SAMPLES_PER_HOUR * 24

    def detect_gaps(self, max_gap_hours: float = 3.0):
        """Detect gaps larger than specified duration."""
        gaps = self.df['time_diff'] > max_gap_hours
        return gaps

    def detect_noise(self, volatility_threshold: float = 0.5, 
                    baseline_multiplier: float = 5.0, 
                    min_amplitude: float = 10.0,
                    window_hours: int = 24):  # Increased to 24 hours
        """Detect noisy periods using daily windows for better context."""
        value_diff = self.df[self.value_column].diff()
        direction_changes = np.sign(value_diff).diff().abs()
        
        # Increase window size for smoother volatility detection
        volatility_window = int(self.SAMPLES_PER_HOUR * window_hours)
        local_volatility = direction_changes.rolling(window=volatility_window, center=True).mean()
        
        # Use larger window for baseline noise calculation
        baseline_window = int(self.SAMPLES_PER_HOUR * window_hours * 2)  # 48 hours for baseline
        baseline_noise = value_diff.abs().rolling(window=baseline_window, center=True).median()
        adaptive_threshold = np.where(baseline_noise > 0.5, baseline_multiplier * baseline_noise, min_amplitude)

        # Add amplitude condition to avoid flagging small fluctuations
        amplitude_condition = abs(value_diff) > min_amplitude

        return (local_volatility > volatility_threshold) & amplitude_condition & (abs(value_diff) > adaptive_threshold)

    def detect_drops(self, std_factor: float = 0.15, 
                    window_hours: int = 48,
                    min_drop: float = 50.0):
        """Detect sustained drops using 2-day windows."""
        window_size = int(self.SAMPLES_PER_HOUR * window_hours)  # Convert to integer
        rolling_min = self.df[self.value_column].rolling(window=window_size).min()
        rolling_max = self.df[self.value_column].rolling(window=window_size).max()
        
        # Calculate local standard deviation
        local_std = self.df[self.value_column].rolling(window=window_size).std()
        std_threshold = local_std * std_factor
        
        # Add minimum drop condition
        drop_magnitude = rolling_max - rolling_min
        magnitude_condition = drop_magnitude > min_drop
        
        # Look for consistent downward trend
        trend_window = int(self.SAMPLES_PER_HOUR * (window_hours // 2))  # Convert to integer
        trend_condition = self.df[self.value_column].diff(trend_window) < -std_threshold
        
        return magnitude_condition & trend_condition

    def detect_point_anomalies(self, min_change: float = 2.02, relative_multiplier: float = 25):
        """Detect point anomalies in the data."""
        value_diff = self.df[self.value_column].diff()
        abs_diff = abs(value_diff)
        
        window_size = int(self.SAMPLES_PER_DAY)  # Use integer window size
        rolling_median = abs_diff.rolling(window=window_size, min_periods=1).median()
        
        spike_mask = (abs_diff > min_change) & (abs_diff > (rolling_median * relative_multiplier))
        
        # Convert float indices to integers when grouping anomaly segments
        anomaly_segments = []
        start_idx = None
        max_gap = int(self.SAMPLES_PER_HOUR // 2)  # Convert to integer
        gap_counter = 0
        
        for i in range(len(spike_mask)):
            if spike_mask[i]:
                if start_idx is None:
                    start_idx = i
                gap_counter = 0
            elif start_idx is not None:
                gap_counter += 1
                if gap_counter > max_gap:
                    segment_start = max(0, int(start_idx - self.SAMPLES_PER_HOUR))
                    segment_end = min(len(self.df), int(i - gap_counter + self.SAMPLES_PER_HOUR))
                    anomaly_segments.append((segment_start, segment_end))
                    start_idx = None
                    gap_counter = 0
        
        if start_idx is not None:
            segment_start = max(0, int(start_idx - self.SAMPLES_PER_HOUR))
            segment_end = min(len(self.df), int(len(spike_mask) + self.SAMPLES_PER_HOUR))
            anomaly_segments.append((segment_start, segment_end))
        
        return anomaly_segments

    def detect_frozen_values(self, min_consecutive: int = 10):
        """Detect frozen values in the data.
        
        Args:
            min_consecutive: Minimum number of consecutive measurements with exactly the same value
        
        Returns:
            pandas.Series: Boolean mask indicating frozen values
        """
        # Initialize mask
        frozen_mask = pd.Series(False, index=self.df.index)
        
        # Get runs of identical values
        value_changes = (self.df[self.value_column] != self.df[self.value_column].shift()).cumsum()
        value_runs = self.df.groupby(value_changes)
        
        # Mark runs that meet the minimum length requirement
        for _, group in value_runs:
            if len(group) >= min_consecutive:
                frozen_mask.iloc[group.index] = True
        
        return frozen_mask

    def detect_offsets(self, window_hours: int = 72, min_offset: float = 50.0, stability_threshold: float = 0.3):
        """Detect offset errors (sudden baseline shifts) in the data.
        
        Args:
            window_hours: Size of windows to compare before/after potential offset
            min_offset: Minimum change in baseline to consider as offset (in mm)
            stability_threshold: How stable the data should be before/after shift (as fraction of offset)
        
        Returns:
            List of tuples containing (start_idx, end_idx, offset_magnitude)
        """
        window_size = int(self.SAMPLES_PER_HOUR * window_hours)  # Convert to integer
        offsets = []
        
        # Calculate rolling statistics
        rolling_mean = self.df[self.value_column].rolling(window=window_size, center=True).mean()
        rolling_std = self.df[self.value_column].rolling(window=window_size, center=True).std()
        
        # Look for significant changes in the mean level
        mean_diff = rolling_mean.diff()
        
        potential_offsets = np.where(abs(mean_diff) > min_offset)[0]
        
        # Group nearby offset points
        if len(potential_offsets) > 0:
            current_group = [potential_offsets[0]]
            
            for i in range(1, len(potential_offsets)):
                if potential_offsets[i] - potential_offsets[i-1] <= window_size:
                    current_group.append(potential_offsets[i])
                else:
                    # Process the group
                    if len(current_group) > 0:
                        mid_point = current_group[len(current_group)//2]
                        
                        # Check stability before and after
                        before_start = max(0, mid_point - window_size)
                        before_end = mid_point
                        after_start = mid_point
                        after_end = min(len(self.df), mid_point + window_size)
                        
                        before_std = self.df[self.value_column].iloc[before_start:before_end].std()
                        after_std = self.df[self.value_column].iloc[after_start:after_end].std()
                        
                        before_mean = self.df[self.value_column].iloc[before_start:before_end].mean()
                        after_mean = self.df[self.value_column].iloc[after_start:after_end].mean()
                        offset_magnitude = after_mean - before_mean
                        
                        # Check if the data is stable enough before and after the shift
                        if (before_std < abs(offset_magnitude) * stability_threshold and 
                            after_std < abs(offset_magnitude) * stability_threshold):
                            offsets.append((before_end, after_start, offset_magnitude))
                    
                    current_group = [potential_offsets[i]]
        
        return offsets