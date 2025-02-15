import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plotly_resampler import FigureResampler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import bisect

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
        self.value_column = value_column
        self.timestamp_column = timestamp_column

        # Ensure the timestamp column exists, or try to find a likely candidate (case-insensitive)
        if timestamp_column not in raw_df.columns:
            possible = [col for col in raw_df.columns if col.lower() == timestamp_column.lower()]
            if possible:
                self.timestamp_column = possible[0]
                print(f"Using column '{self.timestamp_column}' as timestamp.")
            else:
                raise KeyError(
                    f"Timestamp column '{timestamp_column}' not found in raw dataframe. "
                    f"Available columns: {raw_df.columns.tolist()}"
                )

        # Copy the provided dataframes
        self.df = raw_df.copy()
        self.edt_df = edt_df.copy() if edt_df is not None else None
        self.rain_df = rain_df.copy() if rain_df is not None else None
        
        # Process rainfall data if available
        if rain_df is not None:
            self.rain_df = (rain_df
                          .set_index('datetime')
                          .resample('D')['precipitation (mm)']
                          .sum()
                          .reset_index())
        else:
            self.rain_df = None
        
        # Sort by timestamp and reset index
        self.df = self.df.sort_values(self.timestamp_column).reset_index(drop=True)
        if self.edt_df is not None:
            self.edt_df = self.edt_df.sort_values(self.timestamp_column).reset_index(drop=True)
        
        # Convert to a time-indexed DataFrame and use a time-based rolling window
        self.df = self.df.set_index(self.timestamp_column)
        self.df['rolling_mean'] = self.df[self.value_column].rolling("24H", min_periods=1).mean()
        self.df['rolling_std'] = self.df[self.value_column].rolling("24H", min_periods=1).std()
        
        # Calculate time differences and rolling statistics
        self.df['time_diff'] = self.df.index.to_series().diff().dt.total_seconds() / 3600
        
        # Dynamically calculate sampling rate based on median time difference
        if len(self.df) > 1:
            median_diff_hours = self.df.index.to_series().diff().dropna().median().total_seconds() / 3600
            self.SAMPLES_PER_HOUR = 1 / median_diff_hours
        else:
            self.SAMPLES_PER_HOUR = 2.4  # fallback in case there is only one record
        self.SAMPLES_PER_DAY = self.SAMPLES_PER_HOUR * 24

    def detect_gaps(self, min_gap_hours: float = 1):
        """Detect significant gaps in the time series data.
        
        Args:
            min_gap_hours: Minimum gap duration to consider (default: 48.0 hours)
                          Gaps shorter than this will be ignored
        
        Returns:
            pandas.Series: Boolean mask indicating significant gap locations
        """
        # Calculate time differences using the index
        time_diffs = self.df.index.to_series().diff().dt.total_seconds() / 3600
        
        # Initialize gap mask
        gap_mask = pd.Series(False, index=self.df.index)
        
        # Find significant gaps (much larger than normal measurement interval)
        gap_indices = np.where(time_diffs > min_gap_hours)[0]
        
        # Mark gaps and their boundaries
        for idx in gap_indices:
            if idx > 0:  # Skip first row where diff is NaN
                gap_start = self.df.index[idx-1]
                gap_end = self.df.index[idx]
                duration = (gap_end - gap_start).total_seconds() / 3600
                
                # Only mark truly significant gaps
                if duration >= min_gap_hours:
                    # Mark points at both ends of the gap
                    gap_mask.iloc[idx-1:idx+1] = True
        
        return gap_mask

    def detect_noise(self, volatility_threshold: float = 0.5, 
                    baseline_multiplier: float = 5.0, 
                    min_amplitude: float = 10.0,
                    window_hours: int = 24):
        """Detect noisy periods using daily windows for better context."""
        value_diff = self.df[self.value_column].diff()
        direction_changes = np.sign(value_diff).diff().abs()
        
        volatility_window = int(self.SAMPLES_PER_HOUR * window_hours)
        local_volatility = direction_changes.rolling(window=volatility_window, center=True).mean()
        
        baseline_window = int(self.SAMPLES_PER_HOUR * window_hours * 2)
        abs_diff = value_diff.abs()
        mad = abs_diff.rolling(window=baseline_window, center=True).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        adaptive_threshold = baseline_multiplier * mad
        
        amplitude_condition = abs_diff > min_amplitude
        
        return (local_volatility > volatility_threshold) & amplitude_condition & (abs_diff > adaptive_threshold)

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

    def detect_point_anomalies(self, min_change: float = 25.0, relative_multiplier: float = 10, 
                              max_duration_days: float = 1, return_threshold: float = 0.5):
        """Detect point anomalies (spikes) in the data, excluding regions already marked as offsets."""
        # First get offset regions
        offset_segments = self.detect_offsets()
        offset_mask = pd.Series(False, index=self.df.index)
        
        # Mark all offset regions in the mask
        for start_idx, end_idx, _ in offset_segments:
            offset_mask.iloc[start_idx:end_idx+1] = True
        
        # Now detect anomalies only in non-offset regions
        value_diff = self.df[self.value_column].diff()
        abs_diff = value_diff.abs()
        
        window_size = int(self.SAMPLES_PER_DAY)
        rolling_median = abs_diff.rolling(window=window_size, min_periods=1).median()
        
        threshold = np.maximum(min_change, rolling_median * relative_multiplier)
        
        # Only consider points not in offset regions
        spike_mask = (abs_diff > threshold) & (~offset_mask)
        
        anomaly_segments = []
        i = 1
        while i < len(spike_mask)-1:
            if spike_mask.iloc[i] and not offset_mask.iloc[i]:
                # Found potential start of spike
                initial_value = self.df[self.value_column].iloc[i-1]
                spike_value = self.df[self.value_column].iloc[i]
                change = abs(spike_value - initial_value)
                
                # Look ahead for return to original value
                for j in range(i+1, min(i+5, len(self.df))):  # Look up to 4 points ahead
                    return_value = self.df[self.value_column].iloc[j]
                    return_change = abs(return_value - initial_value)
                    
                    # Only consider it a spike if none of the points are in an offset region
                    if (return_change < change * return_threshold and 
                        not offset_mask.iloc[i-1:j+1].any()):
                        # Found return to normal - mark whole segment as spike
                        anomaly_segments.append((i-1, j))
                        i = j  # Skip to end of spike
                        break
                
                i += 1
            else:
                i += 1
        
        return anomaly_segments

    def detect_frozen_values(self, min_consecutive: int = 20):
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
                # Use index locations instead of index values
                frozen_mask.loc[group.index] = True
        
        return frozen_mask

    def detect_offsets(self, min_offset: float = 100.0, min_duration_hours: float = 0.5, 
                      max_duration_days: int = 30, stability_threshold: float = 0.5,
                      merge_window_hours: float = 12):  # New parameter for merging
        """Detect offset errors (sudden baseline shifts) in the data.
        
        Args:
            min_offset: Minimum offset magnitude to consider (mm)
            min_duration_hours: Minimum duration of offset period
            max_duration_days: Maximum duration to consider
            stability_threshold: Maximum allowed variability during offset
            merge_window_hours: Time window within which to merge nearby offsets
        """
        # Calculate value differences and rates of change
        value_diff = self.df[self.value_column].diff()
        abs_diff = value_diff.abs()
        time_diff = self.df.index.to_series().diff().dt.total_seconds() / 3600
        rate_of_change = value_diff / time_diff
        
        # Find significant and sudden changes
        max_natural_rate = 50.0  # mm per hour
        significant_changes = (abs_diff > min_offset) & (abs(rate_of_change) > max_natural_rate)
        change_points = np.where(significant_changes)[0]
        
        offsets = []
        min_samples = max(3, int(self.SAMPLES_PER_HOUR * min_duration_hours))
        max_samples = int(self.SAMPLES_PER_HOUR * 24 * max_duration_days)
        
        i = 0
        while i < len(change_points) - 1:
            start_idx = change_points[i]
            
            # Get the baseline level before the change using a longer window
            baseline_window = min(int(self.SAMPLES_PER_HOUR * 24), start_idx)
            pre_level = self.df[self.value_column].iloc[max(0, start_idx - baseline_window):start_idx].median()
            pre_std = self.df[self.value_column].iloc[max(0, start_idx - baseline_window):start_idx].std()
            
            # Look ahead for a return to baseline
            found_return = False
            for j in range(i + 1, len(change_points)):
                next_idx = change_points[j]
                if next_idx - start_idx < min_samples:
                    continue
                if next_idx - start_idx > max_samples:
                    break
                
                # Check the period between changes
                offset_period = self.df[self.value_column].iloc[start_idx:next_idx]
                offset_level = offset_period.median()
                offset_std = offset_period.std()
                
                # NEW: Check for frozen values
                unique_values = offset_period.nunique()
                has_variation = unique_values > 1  # Must have at least 2 different values
                
                # Look at the level after this period
                post_idx = min(next_idx + min_samples, len(self.df))
                post_level = self.df[self.value_column].iloc[next_idx:post_idx].median()
                
                # Calculate magnitudes
                offset_magnitude = offset_level - pre_level
                return_magnitude = post_level - pre_level
                
                # Additional checks for offset validation
                is_significant = abs(offset_magnitude) > min_offset
                is_stable = offset_std < max(pre_std * 2, abs(offset_magnitude) * stability_threshold)
                is_sustained = next_idx - start_idx >= min_samples
                returns_to_baseline = abs(return_magnitude) < abs(offset_magnitude) * 0.5
                quick_return = next_idx - start_idx < min_samples * 2
                
                # Add has_variation to the validation conditions
                if (is_significant and is_stable and is_sustained and returns_to_baseline 
                    and not quick_return and has_variation):
                    offsets.append((start_idx, next_idx, offset_magnitude))
                    i = j + 1
                    found_return = True
                    break
            
            if not found_return:
                i += 1
        
        # After detecting individual offsets, merge nearby ones
        merged_offsets = []
        if not offsets:
            return merged_offsets
        
        current_group = [offsets[0]]
        
        for offset in offsets[1:]:
            start_idx, end_idx, magnitude = offset
            prev_start_idx, prev_end_idx, prev_magnitude = current_group[-1]
            
            # Calculate time difference between offsets
            time_diff = (self.df.index[start_idx] - self.df.index[prev_end_idx]).total_seconds() / 3600
            
            if time_diff <= merge_window_hours:
                # Add to current group
                current_group.append(offset)
            else:
                # Process current group and start new one
                if len(current_group) > 1:
                    # Merge the group into a single offset
                    group_start_idx = current_group[0][0]
                    group_end_idx = current_group[-1][1]
                    # Calculate the total magnitude as the maximum deviation
                    values = self.df[self.value_column].iloc[group_start_idx:group_end_idx]
                    baseline = self.df[self.value_column].iloc[max(0, group_start_idx-24):group_start_idx].median()
                    total_magnitude = abs(values.median() - baseline)
                    merged_offsets.append((group_start_idx, group_end_idx, total_magnitude))
                else:
                    # Add single offset as is
                    merged_offsets.append(current_group[0])
                current_group = [offset]
        
        # Process final group
        if len(current_group) > 1:
            group_start_idx = current_group[0][0]
            group_end_idx = current_group[-1][1]
            values = self.df[self.value_column].iloc[group_start_idx:group_end_idx]
            baseline = self.df[self.value_column].iloc[max(0, group_start_idx-24):group_start_idx].median()
            total_magnitude = abs(values.median() - baseline)
            merged_offsets.append((group_start_idx, group_end_idx, total_magnitude))
        elif current_group:
            merged_offsets.append(current_group[0])
        
        return merged_offsets

    def summarize_errors(self):
        """
        Aggregate error detection results across multiple error types.
        Returns:
            dict: A summary report with counts and basic magnitude statistics for each error type.
        """
        gap_mask = self.detect_gaps()
        noise_mask = self.detect_noise()
        drops_mask = self.detect_drops()
        anomalies_list = self.detect_point_anomalies()
        frozen_mask = self.detect_frozen_values()
        offsets_list = self.detect_offsets()
        
        gap_durations = []
        gap_indices = np.where(gap_mask)[0]
        for idx in gap_indices:
            if idx > 0:
                gap_duration = (self.df.index[idx] - self.df.index[idx-1]).total_seconds() / 3600
                gap_durations.append(gap_duration)
        
        summary = {
            "gaps": {
                "count": int(gap_mask.sum()),
                "median_duration_hours": np.median(gap_durations) if gap_durations else 0,
            },
            "noise": {
                "count": int(noise_mask.sum())
            },
            "drops": {
                "count": int(drops_mask.sum())
            },
            "point_anomalies": {
                "count": len(anomalies_list),
                "segments": anomalies_list
            },
            "frozen_values": {
                "count": int(frozen_mask.sum())
            },
            "offsets": {
                "count": len(offsets_list),
                "offsets": offsets_list
            }
        }
        return summary

    def detect_drift(self, vinge_data: pd.DataFrame, lower_threshold: float = 10, 
                    upper_threshold: float = 150,
                    max_gap_days: float = 90) -> pd.DataFrame:
        """
        Detect drift periods by comparing sensor data with manual measurements.
        
        Args:
            vinge_data: DataFrame containing manual measurements with 'Date' and 'W.L [cm]' columns
            lower_threshold: Minimum difference to consider as drift (default: 10mm)
            upper_threshold: Maximum difference to consider as drift (default: 150mm)
            max_gap_days: Maximum gap between measurements to consider as same drift period (default: 90 days)
        
        Returns:
            DataFrame containing drift statistics
        """
        # First merge the datasets on the nearest timestamp
        merged_data = pd.merge_asof(
            vinge_data[['Date', 'W.L [cm]']].sort_values('Date'),
            self.df[self.value_column].reset_index(),
            left_on='Date',
            right_on=self.timestamp_column,
            direction='nearest',
            tolerance=pd.Timedelta(minutes=15)
        )
        
        # Calculate the absolute difference between values
        merged_data['difference'] = abs(merged_data[self.value_column] - merged_data['W.L [cm]'])
        
        # Create a boolean mask for points with differences
        has_difference = (merged_data['difference'] > lower_threshold) & (merged_data['difference'] < upper_threshold)
        
        # Get only the drift points
        drift_points = merged_data[has_difference].copy()
        
        if len(drift_points) == 0:
            return pd.DataFrame(columns=[
                'drift_group', 'start_date', 'end_date', 'duration_days',
                'mean_difference', 'max_difference', 'num_points'
            ])
        
        # Sort by date
        drift_points = drift_points.sort_values('Date')
        
        # Initialize drift groups
        drift_points['time_diff'] = drift_points['Date'].diff().dt.total_seconds() / (24 * 3600)  # in days
        drift_points['new_group'] = drift_points['time_diff'] > max_gap_days
        drift_points['drift_group'] = drift_points['new_group'].cumsum()
        
        # Calculate drift statistics for each group
        drift_stats = drift_points.groupby('drift_group').agg({
            'Date': ['min', 'max'],
            'difference': ['mean', 'max', 'count']
        }).reset_index()
        
        # Flatten column names
        drift_stats.columns = [
            'drift_group', 'start_date', 'end_date', 'mean_difference', 
            'max_difference', 'num_points'
        ]
        
        # Calculate duration in days
        drift_stats['duration_days'] = (
            drift_stats['end_date'] - drift_stats['start_date']
        ).dt.total_seconds() / (24 * 3600)
        
        # Reorder columns to match existing format
        drift_stats = drift_stats[[
            'drift_group', 'start_date', 'end_date', 'duration_days',
            'mean_difference', 'max_difference', 'num_points'
        ]]
        
        return drift_stats

    def detect_linear_interpolation(self, min_duration_hours: float = 24.0, 
                                  rate_tolerance: float = 1e-6,
                                  min_points: int = 5,
                                  min_rate: float = 0.001) -> pd.Series:
        """Detect segments of linear interpolation in the data.
        
        Args:
            min_duration_hours: Minimum duration of interpolation to detect (default: 24 hours)
            rate_tolerance: Maximum allowed deviation in rate of change to consider as constant (default: 1e-6)
            min_points: Minimum number of points required to consider a segment (default: 5)
            min_rate: Minimum absolute rate of change to consider (to exclude flat segments) (default: 0.001)
        
        Returns:
            pd.Series: Boolean mask indicating linear interpolation segments
        """
        # Initialize mask for linear segments
        linear_mask = pd.Series(False, index=self.df.index)
        
        # Calculate time differences and value differences
        time_diffs = self.df.index.to_series().diff().dt.total_seconds()
        value_diffs = self.df[self.value_column].diff()
        
        # Calculate rates of change
        rates = value_diffs / time_diffs
        
        # Initialize variables for tracking segments
        current_segment = []
        current_rate = None
        
        # Iterate through the rates to find constant segments
        for idx in range(1, len(rates)):
            if current_segment:
                # Check if current rate matches the segment rate within tolerance
                rate_diff = abs(rates.iloc[idx] - current_rate)
                if rate_diff <= rate_tolerance:
                    current_segment.append(idx)
                else:
                    # Check if the completed segment meets minimum requirements
                    segment_duration = (
                        self.df.index[current_segment[-1]] - 
                        self.df.index[current_segment[0]]
                    ).total_seconds() / 3600
                    
                    # Only mark segments that are:
                    # 1. Long enough
                    # 2. Have enough points
                    # 3. Have a non-zero rate of change
                    if (segment_duration >= min_duration_hours and 
                        len(current_segment) >= min_points and
                        abs(current_rate) >= min_rate):
                        linear_mask.iloc[current_segment] = True
                    
                    # Start new segment
                    current_segment = [idx]
                    current_rate = rates.iloc[idx]
            else:
                # Start new segment
                current_segment = [idx]
                current_rate = rates.iloc[idx]
        
        # Check final segment
        if current_segment:
            segment_duration = (
                self.df.index[current_segment[-1]] - 
                self.df.index[current_segment[0]]
            ).total_seconds() / 3600
            
            if (segment_duration >= min_duration_hours and 
                len(current_segment) >= min_points and
                abs(current_rate) >= min_rate):
                linear_mask.iloc[current_segment] = True
        
        return linear_mask

    def generate_error_statistics(self):
        """Generate comprehensive statistics for each type of error."""
        stats = {}
        total_duration = (self.df.index[-1] - self.df.index[0]).total_seconds() / 3600  # Total hours
        
        # Gap Statistics
        time_diffs = self.df.index.to_series().diff().dt.total_seconds() / 3600
        typical_interval = time_diffs.median()
        gap_threshold = max(typical_interval * 2, 0.5)
        gap_durations = time_diffs[time_diffs > gap_threshold].values
        
        if len(gap_durations) > 0:
            gap_duration_stats = pd.Series(gap_durations).describe(percentiles=[0.25, 0.75])
            stats['gaps'] = {
                'count': len(gap_durations),
                'total_duration_hours': sum(gap_durations),
                'min_duration_hours': gap_duration_stats['min'],
                '25%_duration_hours': gap_duration_stats['25%'],
                'mean_duration_hours': gap_duration_stats['mean'],
                'median_duration_hours': gap_duration_stats['50%'],
                '75%_duration_hours': gap_duration_stats['75%'],
                'max_duration_hours': gap_duration_stats['max'],
                'frequency_percent': (sum(gap_durations) / total_duration) * 100,
                '_raw_durations': gap_durations.tolist()  # Store raw values
            }
        else:
            stats['gaps'] = {
                'count': 0,
                'total_duration_hours': 0,
                'min_duration_hours': 0,
                '25%_duration_hours': 0,
                'mean_duration_hours': 0,
                'median_duration_hours': 0,
                '75%_duration_hours': 0,
                'max_duration_hours': 0,
                'frequency_percent': 0,
                '_raw_durations': []
            }
        
        # Frozen Values Statistics
        frozen = self.detect_frozen_values(min_consecutive=20)
        frozen_periods = []
        current_frozen = None
        prev_idx = None
        
        for idx in frozen[frozen].index:
            if current_frozen is None:
                current_frozen = {'start': idx, 'value': self.df.loc[idx, self.value_column]}
                prev_idx = idx
            elif (idx - prev_idx).total_seconds() > 900:
                if len(frozen[frozen].loc[current_frozen['start']:prev_idx]) >= 20:
                    current_frozen['end'] = prev_idx
                    frozen_periods.append(current_frozen)
                current_frozen = {'start': idx, 'value': self.df.loc[idx, self.value_column]}
            prev_idx = idx
        
        if current_frozen is not None and len(frozen[frozen].loc[current_frozen['start']:prev_idx]) >= 20:
            current_frozen['end'] = prev_idx
            frozen_periods.append(current_frozen)
        
        frozen_durations = []
        for period in frozen_periods:
            duration = (period['end'] - period['start']).total_seconds() / 3600
            if duration > 0:
                frozen_durations.append(duration)
        
        if frozen_durations:
            frozen_duration_stats = pd.Series(frozen_durations).describe(percentiles=[0.25, 0.75])
            stats['frozen_values'] = {
                'count': len(frozen_periods),
                'total_duration_hours': sum(frozen_durations),
                'min_duration_hours': frozen_duration_stats['min'],
                '25%_duration_hours': frozen_duration_stats['25%'],
                'mean_duration_hours': frozen_duration_stats['mean'],
                'median_duration_hours': frozen_duration_stats['50%'],
                '75%_duration_hours': frozen_duration_stats['75%'],
                'max_duration_hours': frozen_duration_stats['max'],
                'frequency_percent': (sum(frozen_durations) / total_duration) * 100,
                '_raw_durations': frozen_durations  # Store raw values
            }
        else:
            stats['frozen_values'] = {
                'count': 0,
                'total_duration_hours': 0,
                'min_duration_hours': 0,
                '25%_duration_hours': 0,
                'mean_duration_hours': 0,
                'median_duration_hours': 0,
                '75%_duration_hours': 0,
                'max_duration_hours': 0,
                'frequency_percent': 0,
                '_raw_durations': []
            }
        
        # Point Anomalies Statistics
        anomalies = self.detect_point_anomalies(
            min_change=25.0,
            relative_multiplier=10,
            max_duration_days=1,
            return_threshold=0.5
        )
        
        anomaly_magnitudes = []
        anomaly_durations = []
        
        for start_idx, end_idx in anomalies:
            initial_value = self.df[self.value_column].iloc[start_idx]
            anomaly_sequence = self.df[self.value_column].iloc[start_idx:end_idx+1]
            magnitude = max(abs(anomaly_sequence - initial_value))
            duration = (self.df.index[end_idx] - self.df.index[start_idx]).total_seconds() / 3600
            
            anomaly_magnitudes.append(magnitude)
            anomaly_durations.append(duration)
        
        if anomaly_magnitudes:
            magnitude_stats = pd.Series(anomaly_magnitudes).describe(percentiles=[0.25, 0.75])
            duration_stats = pd.Series(anomaly_durations).describe(percentiles=[0.25, 0.75])
            stats['point_anomalies'] = {
                'count': len(anomalies),
                'total_duration_hours': sum(anomaly_durations),
                'min_duration_hours': duration_stats['min'],
                '25%_duration_hours': duration_stats['25%'],
                'mean_duration_hours': duration_stats['mean'],
                'median_duration_hours': duration_stats['50%'],
                '75%_duration_hours': duration_stats['75%'],
                'max_duration_hours': duration_stats['max'],
                'min_magnitude_mm': magnitude_stats['min'],
                '25%_magnitude_mm': magnitude_stats['25%'],
                'mean_magnitude_mm': magnitude_stats['mean'],
                'median_magnitude_mm': magnitude_stats['50%'],
                '75%_magnitude_mm': magnitude_stats['75%'],
                'max_magnitude_mm': magnitude_stats['max'],
                'frequency_percent': (sum(anomaly_durations) / total_duration) * 100,
                '_raw_magnitudes': anomaly_magnitudes,  # Store raw values
                '_raw_durations': anomaly_durations
            }
        else:
            stats['point_anomalies'] = {
                'count': 0,
                'total_duration_hours': 0,
                'min_duration_hours': 0,
                '25%_duration_hours': 0,
                'mean_duration_hours': 0,
                'median_duration_hours': 0,
                '75%_duration_hours': 0,
                'max_duration_hours': 0,
                'min_magnitude_mm': 0,
                '25%_magnitude_mm': 0,
                'mean_magnitude_mm': 0,
                'median_magnitude_mm': 0,
                '75%_magnitude_mm': 0,
                'max_magnitude_mm': 0,
                'frequency_percent': 0,
                '_raw_magnitudes': [],
                '_raw_durations': []
            }
        
        # Offset Statistics
        offsets = self.detect_offsets()
        offset_magnitudes = []
        offset_durations = []
        
        for start_idx, end_idx, magnitude in offsets:
            duration = (self.df.index[end_idx] - self.df.index[start_idx]).total_seconds() / 3600
            offset_magnitudes.append(abs(magnitude))
            offset_durations.append(duration)
        
        if offset_magnitudes:
            magnitude_stats = pd.Series(offset_magnitudes).describe(percentiles=[0.25, 0.75])
            duration_stats = pd.Series(offset_durations).describe(percentiles=[0.25, 0.75])
            stats['offsets'] = {
                'count': len(offsets),
                'total_duration_hours': sum(offset_durations),
                'min_duration_hours': duration_stats['min'],
                '25%_duration_hours': duration_stats['25%'],
                'mean_duration_hours': duration_stats['mean'],
                'median_duration_hours': duration_stats['50%'],
                '75%_duration_hours': duration_stats['75%'],
                'max_duration_hours': duration_stats['max'],
                'min_magnitude_mm': magnitude_stats['min'],
                '25%_magnitude_mm': magnitude_stats['25%'],
                'mean_magnitude_mm': magnitude_stats['mean'],
                'median_magnitude_mm': magnitude_stats['50%'],
                '75%_magnitude_mm': magnitude_stats['75%'],
                'max_magnitude_mm': magnitude_stats['max'],
                'frequency_percent': (sum(offset_durations) / total_duration) * 100,
                '_raw_magnitudes': offset_magnitudes,  # Store raw values
                '_raw_durations': offset_durations
            }
        else:
            stats['offsets'] = {
                'count': 0,
                'total_duration_hours': 0,
                'min_duration_hours': 0,
                '25%_duration_hours': 0,
                'mean_duration_hours': 0,
                'median_duration_hours': 0,
                '75%_duration_hours': 0,
                'max_duration_hours': 0,
                'min_magnitude_mm': 0,
                '25%_magnitude_mm': 0,
                'mean_magnitude_mm': 0,
                'median_magnitude_mm': 0,
                '75%_magnitude_mm': 0,
                'max_magnitude_mm': 0,
                'frequency_percent': 0,
                '_raw_magnitudes': [],
                '_raw_durations': []
            }
        
        return stats

    def save_error_statistics(self, folder: str, output_dir: Path):
        """
        Save error statistics to a formatted Excel file.
        
        Args:
            folder: Station folder name/ID
            output_dir: Directory to save the Excel file
        """
        stats = self.generate_error_statistics()
        
        # Create a pandas ExcelWriter object
        output_path = output_dir / f'error_statistics_{folder}.xlsx'
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Convert the nested dictionary to a more Excel-friendly format
            summary_data = []
            
            for error_type, error_stats in stats.items():
                # Add a row for each statistic
                for stat_name, value in error_stats.items():
                    if isinstance(value, (int, float)):
                        summary_data.append({
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': stat_name.replace('_', ' ').title(),
                            'Value': value
                        })
            
            # Create DataFrame and write to Excel
            df = pd.DataFrame(summary_data)
            df.to_excel(writer, sheet_name='Error Statistics', index=False)
            
            # Get workbook and worksheet objects for formatting
            workbook = writer.book
            worksheet = writer.sheets['Error Statistics']
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'font_size': 12,
                'bg_color': '#4F81BD',
                'font_color': 'white',
                'border': 1
            })
            
            cell_format = workbook.add_format({
                'font_size': 11,
                'border': 1
            })
            
            number_format = workbook.add_format({
                'font_size': 11,
                'border': 1,
                'num_format': '0.00'
            })
            
            # Set column widths
            worksheet.set_column('A:A', 20)  # Error Type
            worksheet.set_column('B:B', 25)  # Statistic
            worksheet.set_column('C:C', 15)  # Value
            
            # Add headers
            for col_num, value in enumerate(['Error Type', 'Statistic', 'Value']):
                worksheet.write(0, col_num, value, header_format)
            
            # Write data with formatting
            for row_num in range(len(df)):
                worksheet.write(row_num + 1, 0, df.iloc[row_num, 0], cell_format)
                worksheet.write(row_num + 1, 1, df.iloc[row_num, 1], cell_format)
                worksheet.write(row_num + 1, 2, df.iloc[row_num, 2], number_format)
            
            # Add autofilter
            worksheet.autofilter(0, 0, len(df), 2)

    @staticmethod
    def save_combined_statistics(all_stats: list[dict], folders: list[str], output_dir: Path):
        """Save all statistics to a single Excel file with multiple sheets."""
        output_path = output_dir / 'error_statistics.xlsx'
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'font_size': 12,
                'bg_color': '#4F81BD',
                'font_color': 'white',
                'border': 1
            })
            
            cell_format = workbook.add_format({
                'font_size': 11,
                'border': 1
            })
            
            number_format = workbook.add_format({
                'font_size': 11,
                'border': 1,
                'num_format': '0.00'
            })
            
            percent_format = workbook.add_format({
                'font_size': 11,
                'border': 1,
                'num_format': '0.00%'
            })
            
            # Collect all raw values for combined statistics
            combined_raw_data = {
                'point_anomalies': {'magnitudes': [], 'durations': []},
                'offsets': {'magnitudes': [], 'durations': []},
                'gaps': {'durations': []},
                'frozen_values': {'durations': []},
                'linear_interpolation': {'durations': []}
            }
            
            # Collect total counts and durations
            total_counts = {error_type: 0 for error_type in combined_raw_data.keys()}
            total_durations = {error_type: 0 for error_type in combined_raw_data.keys()}
            
            # Combine raw data from all stations
            for stats in all_stats:
                for error_type in combined_raw_data.keys():
                    if error_type in stats:
                        # Add raw magnitudes if available
                        if '_raw_magnitudes' in stats[error_type]:
                            combined_raw_data[error_type]['magnitudes'].extend(
                                stats[error_type]['_raw_magnitudes'])
                        
                        # Add raw durations
                        if '_raw_durations' in stats[error_type]:
                            combined_raw_data[error_type]['durations'].extend(
                                stats[error_type]['_raw_durations'])
                        
                        # Update totals
                        total_counts[error_type] += stats[error_type]['count']
                        total_durations[error_type] += stats[error_type]['total_duration_hours']
            
            # Calculate combined statistics
            summary_data = []
            for error_type, raw_data in combined_raw_data.items():
                # Basic statistics for all error types
                summary_data.append({
                    'Error Type': error_type.replace('_', ' ').title(),
                    'Statistic': 'Total Count',
                    'Value': total_counts[error_type]
                })
                
                summary_data.append({
                    'Error Type': error_type.replace('_', ' ').title(),
                    'Statistic': 'Total Duration (hours)',
                    'Value': total_durations[error_type]
                })
                
                # Duration statistics
                if raw_data['durations']:
                    duration_stats = pd.Series(raw_data['durations']).describe(
                        percentiles=[0.25, 0.75])
                    
                    summary_data.extend([
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': 'Min Duration (hours)',
                            'Value': duration_stats['min']
                        },
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': '25% Duration (hours)',
                            'Value': duration_stats['25%']
                        },
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': 'Mean Duration (hours)',
                            'Value': duration_stats['mean']
                        },
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': 'Median Duration (hours)',
                            'Value': duration_stats['50%']
                        },
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': '75% Duration (hours)',
                            'Value': duration_stats['75%']
                        },
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': 'Max Duration (hours)',
                            'Value': duration_stats['max']
                        }
                    ])
                
                # Magnitude statistics only for point anomalies and offsets
                if error_type in ['point_anomalies', 'offsets'] and raw_data['magnitudes']:
                    magnitude_stats = pd.Series(raw_data['magnitudes']).describe(
                        percentiles=[0.25, 0.75])
                    
                    summary_data.extend([
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': 'Min Magnitude (mm)',
                            'Value': magnitude_stats['min']
                        },
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': '25% Magnitude (mm)',
                            'Value': magnitude_stats['25%']
                        },
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': 'Mean Magnitude (mm)',
                            'Value': magnitude_stats['mean']
                        },
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': 'Median Magnitude (mm)',
                            'Value': magnitude_stats['50%']
                        },
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': '75% Magnitude (mm)',
                            'Value': magnitude_stats['75%']
                        },
                        {
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': 'Max Magnitude (mm)',
                            'Value': magnitude_stats['max']
                        }
                    ])
            
            # Create and write combined statistics DataFrame
            combined_df = pd.DataFrame(summary_data)
            combined_df.to_excel(writer, sheet_name='Combined Statistics', index=False)
            
            # Format combined statistics sheet
            worksheet = writer.sheets['Combined Statistics']
            for col_num, value in enumerate(['Error Type', 'Statistic', 'Value']):
                worksheet.write(0, col_num, value, header_format)
            
            worksheet.set_column('A:A', 20)
            worksheet.set_column('B:B', 25)
            worksheet.set_column('C:C', 15)
            
            # Write data with formatting
            for row_num in range(len(combined_df)):
                worksheet.write(row_num + 1, 0, combined_df.iloc[row_num, 0], cell_format)
                worksheet.write(row_num + 1, 1, combined_df.iloc[row_num, 1], cell_format)
                
                value = combined_df.iloc[row_num, 2]
                if 'frequency' in combined_df.iloc[row_num, 1].lower():
                    worksheet.write(row_num + 1, 2, value/100, percent_format)
                else:
                    worksheet.write(row_num + 1, 2, value, number_format)
            
            # Write individual station sheets
            for folder, stats in zip(folders, all_stats):
                sheet_name = f'Station {folder}'
                summary_data = []
                
                for error_type, error_stats in stats.items():
                    for stat_name, value in error_stats.items():
                        if isinstance(value, (int, float)) and not stat_name.startswith('_'):
                            summary_data.append({
                                'Error Type': error_type.replace('_', ' ').title(),
                                'Statistic': stat_name.replace('_', ' ').title(),
                                'Value': value
                            })
                
                df = pd.DataFrame(summary_data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Format station sheet
                worksheet = writer.sheets[sheet_name]
                for col_num, value in enumerate(['Error Type', 'Statistic', 'Value']):
                    worksheet.write(0, col_num, value, header_format)
                
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:B', 25)
                worksheet.set_column('C:C', 15)
                
                for row_num in range(len(df)):
                    worksheet.write(row_num + 1, 0, df.iloc[row_num, 0], cell_format)
                    worksheet.write(row_num + 1, 1, df.iloc[row_num, 1], cell_format)
                    
                    value = df.iloc[row_num, 2]
                    if 'frequency' in df.iloc[row_num, 1].lower():
                        worksheet.write(row_num + 1, 2, value/100, percent_format)
                    else:
                        worksheet.write(row_num + 1, 2, value, number_format)
                
                worksheet.autofilter(0, 0, len(df), 2)