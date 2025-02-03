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

    def detect_gaps(self, min_gap_hours: float = 48.0):
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

    def detect_point_anomalies(self, min_change: float = 500.0, relative_multiplier: float = 5, 
                              max_duration_days: int = 14):
        """Detect point anomalies (spikes) in the data."""
        # Calculate the first difference in the water level values
        value_diff = self.df[self.value_column].diff()
        abs_diff = value_diff.abs()
        
        # Use a rolling window to compute a local median
        window_size = int(self.SAMPLES_PER_DAY)
        rolling_median = abs_diff.rolling(window=window_size, min_periods=1).median()
        
        # Create a combined threshold
        threshold = np.maximum(min_change, rolling_median * relative_multiplier)
        
        # Flag points where the absolute difference exceeds our threshold
        spike_mask = abs_diff > threshold
        
        # Group successive anomaly points into segments
        anomaly_segments = []
        start_idx = None
        timestamps = self.df.index
        max_duration = pd.Timedelta(days=max_duration_days)
        
        for i in range(1, len(spike_mask)):
            if spike_mask.iloc[i]:
                if start_idx is None:
                    start_idx = i - 1  # Include the point before the spike
                elif timestamps[i] - timestamps[start_idx] > max_duration:
                    start_idx = None
            else:
                if start_idx is not None:
                    duration = timestamps[i] - timestamps[start_idx]
                    if duration <= max_duration:
                        anomaly_segments.append((start_idx, i))
                    start_idx = None
        
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
                      max_duration_days: int = 30, stability_threshold: float = 0.5):
        """Detect offset errors (sudden baseline shifts) in the data.
        
        An offset error is characterized by:
        1. A sudden jump/drop from the normal baseline
        2. A period of normal variability at the new offset level (at least 30 minutes)
        3. Eventually returning to approximately the original baseline
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
        
        return offsets

    def analyze_data_characteristics(self):
        """Analyze statistical characteristics of the water level data at multiple time scales."""
        # Calculate time differences and value changes
        time_diffs = self.df.index.to_series().diff().dt.total_seconds() / 3600
        value_diffs = self.df[self.value_column].diff()
        rates = value_diffs / time_diffs
        
        # Basic sampling statistics
        print("\nData Characteristics Summary:")
        print(f"Total duration: {(self.df.index.max() - self.df.index.min()).days} days")
        print(f"Typical sampling interval: {time_diffs.median():.2f} hours")
        
        # Short-term changes (hourly)
        print("\nShort-term Changes (Hourly):")
        hourly_stats = self.df.resample('H')[self.value_column].agg(['min', 'max', 'std'])
        hourly_range = hourly_stats['max'] - hourly_stats['min']
        print(f"Median hourly range: {hourly_range.median():.2f} mm")
        print(f"95th percentile hourly range: {hourly_range.quantile(0.95):.2f} mm")
        print(f"Typical hourly std: {hourly_stats['std'].median():.2f} mm")
        
        # Daily patterns
        print("\nDaily Patterns:")
        daily_stats = self.df.resample('D')[self.value_column].agg(['min', 'max', 'std'])
        daily_range = daily_stats['max'] - daily_stats['min']
        print(f"Median daily range: {daily_range.median():.2f} mm")
        print(f"95th percentile daily range: {daily_range.quantile(0.95):.2f} mm")
        print(f"Maximum daily range: {daily_range.max():.2f} mm")
        print(f"Typical daily std: {daily_stats['std'].median():.2f} mm")
        
        # Weekly patterns
        print("\nWeekly Patterns:")
        weekly_stats = self.df.resample('W')[self.value_column].agg(['min', 'max', 'std'])
        weekly_range = weekly_stats['max'] - weekly_stats['min']
        print(f"Median weekly range: {weekly_range.median():.2f} mm")
        print(f"95th percentile weekly range: {weekly_range.quantile(0.95):.2f} mm")
        print(f"Typical weekly std: {weekly_stats['std'].median():.2f} mm")
        
        # Monthly patterns
        print("\nMonthly Patterns:")
        monthly_stats = self.df.resample('M')[self.value_column].agg(['min', 'max', 'std'])
        monthly_range = monthly_stats['max'] - monthly_stats['min']
        print(f"Median monthly range: {monthly_range.median():.2f} mm")
        print(f"95th percentile monthly range: {monthly_range.quantile(0.95):.2f} mm")
        print(f"Typical monthly std: {monthly_stats['std'].median():.2f} mm")
        
        # Rate of change statistics
        print("\nRate of Change Statistics:")
        print(f"Median absolute rate: {abs(rates).median():.2f} mm/hour")
        print(f"95th percentile rate: {abs(rates).quantile(0.95):.2f} mm/hour")
        print(f"99th percentile rate: {abs(rates).quantile(0.99):.2f} mm/hour")
        
        # Extreme changes
        print("\nExtreme Changes:")
        print(f"Maximum rise: {value_diffs.max():.2f} mm")
        print(f"Maximum fall: {value_diffs.min():.2f} mm")
        print(f"Maximum rate of rise: {rates.max():.2f} mm/hour")
        print(f"Maximum rate of fall: {rates.min():.2f} mm/hour")

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
                    upper_threshold: float = 150) -> pd.DataFrame:
        """
        Detect drift periods by comparing sensor data with manual measurements.
        
        Args:
            vinge_data: DataFrame containing manual measurements with 'Date' and 'W.L [cm]' columns
            lower_threshold: Minimum difference to consider as drift (default: 10mm)
            upper_threshold: Maximum difference to consider as drift (default: 150mm)
        
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
        
        # Create groups of continuous drift periods
        merged_data['drift_group'] = (~has_difference).cumsum()[has_difference]
        
        # Calculate drift statistics
        drift_stats = merged_data[has_difference].groupby('drift_group').agg({
            'Date': ['min', 'max', lambda x: (x.max() - x.min()).total_seconds() / (60 * 60 * 24)],
            'difference': ['mean', 'max', 'count']
        }).reset_index()
        
        # Rename columns for clarity
        drift_stats.columns = [
            'drift_group', 'start_date', 'end_date', 'duration_days',
            'mean_difference', 'max_difference', 'num_points'
        ]
        
        return drift_stats

    def generate_error_statistics(self):
        """Generate comprehensive statistics for each type of error."""
        stats = {}
        total_duration = (self.df.index[-1] - self.df.index[0]).total_seconds() / 3600  # Total hours
        
        # Gap Statistics
        gaps = self.detect_gaps()
        gap_periods = []
        current_gap = None
        
        for idx in gaps[gaps].index:
            if current_gap is None:
                current_gap = {'start': idx}
            elif (idx - prev_idx).total_seconds() > 3600:  # New gap if > 1 hour between points
                current_gap['end'] = prev_idx
                gap_periods.append(current_gap)
                current_gap = {'start': idx}
            prev_idx = idx
        
        if current_gap is not None:
            current_gap['end'] = prev_idx
            gap_periods.append(current_gap)
        
        gap_durations = [(gap['end'] - gap['start']).total_seconds() / 3600 for gap in gap_periods]
        stats['gaps'] = {
            'count': len(gap_periods),
            'total_duration_hours': sum(gap_durations),
            'avg_duration_hours': np.mean(gap_durations) if gap_durations else 0,
            'max_duration_hours': max(gap_durations) if gap_durations else 0,
            'frequency_percent': (sum(gap_durations) / total_duration) * 100 if gap_durations else 0
        }
        
        # Frozen Values Statistics
        frozen = self.detect_frozen_values()
        frozen_periods = []
        current_frozen = None
        
        for idx in frozen[frozen].index:
            if current_frozen is None:
                current_frozen = {'start': idx, 'value': self.df.loc[idx, self.value_column]}
            elif (idx - prev_idx).total_seconds() > 3600:  # New period if > 1 hour between points
                current_frozen['end'] = prev_idx
                frozen_periods.append(current_frozen)
                current_frozen = {'start': idx, 'value': self.df.loc[idx, self.value_column]}
            prev_idx = idx
        
        if current_frozen is not None:
            current_frozen['end'] = prev_idx
            frozen_periods.append(current_frozen)
        
        frozen_durations = [(period['end'] - period['start']).total_seconds() / 3600 for period in frozen_periods]
        stats['frozen_values'] = {
            'count': len(frozen_periods),
            'total_duration_hours': sum(frozen_durations),
            'avg_duration_hours': np.mean(frozen_durations) if frozen_durations else 0,
            'max_duration_hours': max(frozen_durations) if frozen_durations else 0,
            'frequency_percent': (sum(frozen_durations) / total_duration) * 100 if frozen_durations else 0
        }
        
        # Point Anomalies Statistics
        anomalies = self.detect_point_anomalies()
        anomaly_magnitudes = []
        anomaly_durations = []
        
        for start_idx, end_idx in anomalies:
            segment = self.df.iloc[start_idx:end_idx+1]
            baseline = self.df[self.value_column].iloc[max(0, start_idx-10):start_idx].mean()
            magnitude = abs(segment[self.value_column] - baseline).max()
            duration = (segment.index[-1] - segment.index[0]).total_seconds() / 3600
            
            anomaly_magnitudes.append(magnitude)
            anomaly_durations.append(duration)
        
        stats['point_anomalies'] = {
            'count': len(anomalies),
            'total_duration_hours': sum(anomaly_durations),
            'avg_duration_hours': np.mean(anomaly_durations) if anomaly_durations else 0,
            'avg_magnitude_mm': np.mean(anomaly_magnitudes) if anomaly_magnitudes else 0,
            'max_magnitude_mm': max(anomaly_magnitudes) if anomaly_magnitudes else 0,
            'frequency_percent': (sum(anomaly_durations) / total_duration) * 100 if anomaly_durations else 0
        }
        
        # Offset Statistics
        offsets = self.detect_offsets()
        offset_magnitudes = []
        offset_durations = []
        
        for start_idx, end_idx, magnitude in offsets:
            duration = (self.df.index[end_idx] - self.df.index[start_idx]).total_seconds() / 3600
            offset_magnitudes.append(abs(magnitude))
            offset_durations.append(duration)
        
        stats['offsets'] = {
            'count': len(offsets),
            'total_duration_hours': sum(offset_durations),
            'avg_duration_hours': np.mean(offset_durations) if offset_durations else 0,
            'avg_magnitude_mm': np.mean(offset_magnitudes) if offset_magnitudes else 0,
            'max_magnitude_mm': max(offset_magnitudes) if offset_magnitudes else 0,
            'frequency_percent': (sum(offset_durations) / total_duration) * 100 if offset_durations else 0
        }
        
        # Drift Statistics
        if hasattr(self, 'drift_stats') and self.drift_stats is not None:
            drift_magnitudes = self.drift_stats['mean_difference'].values
            drift_durations = self.drift_stats['duration_days'].values * 24
            
            stats['drift'] = {
                'count': len(self.drift_stats),
                'total_duration_hours': sum(drift_durations),
                'avg_duration_hours': np.mean(drift_durations) if len(drift_durations) > 0 else 0,
                'avg_magnitude_mm': np.mean(drift_magnitudes) if len(drift_magnitudes) > 0 else 0,
                'max_magnitude_mm': max(drift_magnitudes) if len(drift_magnitudes) > 0 else 0,
                'frequency_percent': (sum(drift_durations) / total_duration) * 100 if len(drift_durations) > 0 else 0
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
            
            # Define common formats
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
            
            # First, create the combined statistics sheet
            combined_stats = {}
            for stats in all_stats:
                for error_type, error_stats in stats.items():
                    if error_type not in combined_stats:
                        combined_stats[error_type] = {stat: [] for stat in error_stats.keys()}
                    for stat, value in error_stats.items():
                        combined_stats[error_type][stat].append(value)
            
            summary_data = []
            for error_type, error_stats in combined_stats.items():
                for stat, values in error_stats.items():
                    if values:
                        summary_data.append({
                            'Error Type': error_type.replace('_', ' ').title(),
                            'Statistic': stat.replace('_', ' ').title(),
                            'Average': np.mean(values),
                            'Min': np.min(values),
                            'Max': np.max(values),
                            'Total': np.sum(values) if 'count' in stat.lower() or 'duration' in stat.lower() else None
                        })
            
            combined_df = pd.DataFrame(summary_data)
            combined_df.to_excel(writer, sheet_name='Combined Statistics', index=False)
            
            # Format combined statistics sheet
            worksheet = writer.sheets['Combined Statistics']
            for col_num, value in enumerate(['Error Type', 'Statistic', 'Average', 'Min', 'Max', 'Total']):
                worksheet.write(0, col_num, value, header_format)
            worksheet.set_column('A:A', 20)
            worksheet.set_column('B:B', 25)
            worksheet.set_column('C:F', 15)
            
            # Then create individual station sheets
            for folder, stats in zip(folders, all_stats):
                sheet_name = f'Station {folder}'
                summary_data = []
                
                for error_type, error_stats in stats.items():
                    for stat_name, value in error_stats.items():
                        if isinstance(value, (int, float)):
                            row_data = {
                                'Error Type': error_type.replace('_', ' ').title(),
                                'Statistic': stat_name.replace('_', ' ').title(),
                                'Value': value
                            }
                            summary_data.append(row_data)
                
                df = pd.DataFrame(summary_data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Format station sheet
                worksheet = writer.sheets[sheet_name]
                for col_num, value in enumerate(['Error Type', 'Statistic', 'Value']):
                    worksheet.write(0, col_num, value, header_format)
                
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:B', 25)
                worksheet.set_column('C:C', 15)
                
                # Write data with appropriate formatting
                for row_num in range(len(df)):
                    worksheet.write(row_num + 1, 0, df.iloc[row_num, 0], cell_format)
                    worksheet.write(row_num + 1, 1, df.iloc[row_num, 1], cell_format)
                    
                    # Use percent format for frequency statistics
                    value = df.iloc[row_num, 2]
                    if 'frequency' in df.iloc[row_num, 1].lower():
                        worksheet.write(row_num + 1, 2, value/100, percent_format)
                    else:
                        worksheet.write(row_num + 1, 2, value, number_format)
                
                worksheet.autofilter(0, 0, len(df), 2)