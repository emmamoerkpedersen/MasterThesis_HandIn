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

    def plot_all_errors(self):
        """Plot all error types using the detection functions with plotly-resampler."""
        # Create subplot figure with secondary y-axes
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Linear Interpolation Detection', 'Noise Detection',
                'Sustained Drop Detection', 'Point Anomaly Detection',
                'Frozen Value Detection', 'Offset Detection'
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Create a FigureResampler object
        fig = FigureResampler(fig)
        
        # First add edited data (in background)
        if self.edt_df is not None:
            for row in range(1, 4):
                for col in range(1, 3):
                    fig.add_trace(
                        go.Scattergl(
                            name='Edited Data',
                            line=dict(color='red', width=1, dash='dash'),
                            showlegend=bool(row == 1 and col == 1),
                            hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
                        ),
                        hf_x=self.edt_df[self.timestamp_column],
                        hf_y=self.edt_df[self.value_column],
                        row=row, col=col,
                        secondary_y=False
                    )
        
        # Add original data
        for row in range(1, 4):
            for col in range(1, 3):
                fig.add_trace(
                    go.Scattergl(
                        name='Original Data',
                        showlegend=bool(row == 1 and col == 1),
                        line=dict(color='black', width=1),
                        hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
                    ),
                    hf_x=self.df[self.timestamp_column],
                    hf_y=self.df[self.value_column],
                    row=row, col=col,
                    secondary_y=False
                )
        
        # Add rainfall data after the resampled traces
        if self.rain_df is not None:
            # Calculate the y-axis range for the water level data
            y_min = self.df[self.value_column].min()
            y_max = self.df[self.value_column].max()
            y_range = y_max - y_min
            
            # Place bars starting from the top of the plot
            bar_base = y_max + (y_range * 0.05)
            
            # Get max rainfall value
            max_rainfall = self.rain_df['precipitation (mm)'].max()
            
            # Scale the bars to extend about halfway down
            scaled_rainfall = -(self.rain_df['precipitation (mm)'])  # No scaling factor, just negate
            
            fig.add_trace(
                go.Bar(
                    name='Daily Rainfall',
                    x=self.rain_df['datetime'],
                    y=scaled_rainfall,  # Use negated values
                    marker_color='blue',
                    opacity=0.3,
                    width=24*60*60*1000,  # Set bar width to 1 day in milliseconds
                    showlegend=True,
                    base=bar_base,
                    hovertemplate='Date: %{x}<br>Daily Rainfall: %{customdata:.1f} mm<extra></extra>',
                    customdata=self.rain_df['precipitation (mm)']
                ),
                secondary_y=True  # Changed to use secondary y-axis
            )
            
            # Update primary y-axis for water level data
            fig.update_yaxes(
                title_text="Water Level (mm)",
                title_font=dict(size=16),
                secondary_y=False,
                range=[y_min - (y_range * 0.05), bar_base + (y_range * 0.05)]
            )
            
            # Update secondary y-axis for rainfall with doubled range
            fig.update_yaxes(
                title_text="Daily Rainfall (mm)",
                secondary_y=True,
                range=[max_rainfall * 2, 0],  # Double the range, reversed scale
                overlaying="y"
            )
        
        # 1. Linear interpolation (row=1, col=1)
        gaps = self.detect_gaps()
        if any(gaps):
            gap_periods = self.df[gaps].index
            first_gap = True
            for idx in gap_periods:
                if idx+1 < len(self.df):
                    # Add vertical lines at gap boundaries
                    for x in [self.df[self.timestamp_column].iloc[idx], 
                             self.df[self.timestamp_column].iloc[idx+1]]:
                        fig.add_trace(
                            go.Scatter(
                                x=[x, x],
                                y=[self.df[self.value_column].min(), self.df[self.value_column].max()],
                                mode='lines',
                                line=dict(color='purple', width=1),
                                name='Linear Interpolation',
                                showlegend=bool(first_gap)  # Convert to Python bool
                            ),
                            row=1, col=1
                        )
                    first_gap = False
        
        # 2. Noise detection (row=1, col=2)
        noisy_idx = self.detect_noise()
        if any(noisy_idx):
            # Find continuous periods of noise
            noise_changes = (noisy_idx != noisy_idx.shift()).cumsum()
            for period in noise_changes.unique():
                period_mask = noise_changes == period
                if noisy_idx[period_mask].any():
                    start_date = self.df[self.timestamp_column][period_mask].iloc[0]
                    end_date = self.df[self.timestamp_column][period_mask].iloc[-1]
                    for x in [start_date, end_date]:
                        fig.add_trace(
                            go.Scatter(
                                x=[x, x],
                                y=[self.df[self.value_column].min(), self.df[self.value_column].max()],
                                mode='lines',
                                line=dict(color='yellow', width=1),
                                name='Noisy Period',
                                showlegend=bool(period == noise_changes.unique()[0])  # Convert to Python bool
                            ),
                            row=1, col=2
                        )
        
        # 3. Drops (row=2, col=1)
        sustained_drops = self.detect_drops()
        if any(sustained_drops):
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.timestamp_column][sustained_drops],
                    y=self.df[self.value_column][sustained_drops],
                    mode='lines',
                    line=dict(color='green', width=2),
                    name='Sustained Drops',
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # 4. Point anomalies (row=2, col=2)
        anomaly_segments = self.detect_point_anomalies()
        first_anomaly = True
        for start, end in anomaly_segments:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.timestamp_column].iloc[start:end],
                    y=self.df[self.value_column].iloc[start:end],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Point Anomalies',
                    showlegend=first_anomaly
                ),
                row=2, col=2
            )
            first_anomaly = False
        
        # 5. Frozen values (row=3, col=1)
        frozen_mask = self.detect_frozen_values()
        if any(frozen_mask):
            # Group consecutive frozen periods
            frozen_groups = (frozen_mask != frozen_mask.shift()).cumsum()
            first_frozen = True
            
            for group in frozen_groups.unique():
                if frozen_mask[frozen_groups == group].any():
                    group_mask = frozen_groups == group
                    start_date = self.df[self.timestamp_column][group_mask].iloc[0]
                    end_date = self.df[self.timestamp_column][group_mask].iloc[-1]
                    frozen_value = self.df[self.value_column][group_mask].iloc[0]
                    n_measurements = sum(group_mask)
                    
                    # Add vertical lines at start and end of frozen period
                    for x in [start_date, end_date]:
                        fig.add_trace(
                            go.Scatter(
                                x=[x, x],
                                y=[self.df[self.value_column].min(), self.df[self.value_column].max()],
                                mode='lines',
                                line=dict(color='red', width=2),
                                name='Frozen Values',
                                showlegend=bool(first_frozen),  # Convert to Python bool
                                hovertemplate=(
                                    'Date: %{x}<br>'
                                    f'Frozen Value: {frozen_value:.2f} mm<br>'
                                    f'Consecutive measurements: {n_measurements}<extra></extra>'
                                )
                            ),
                            row=3, col=1
                        )
                    first_frozen = False
        
        # 6. Offsets (row=3, col=2)
        offsets = self.detect_offsets()
        first_offset = True
        if offsets:
            for start, end, magnitude in offsets:
                fig.add_trace(
                    go.Scatter(
                        x=[self.df[self.timestamp_column].iloc[start], 
                           self.df[self.timestamp_column].iloc[end]],
                        y=[self.df[self.value_column].iloc[start], 
                           self.df[self.value_column].iloc[end]],
                        mode='lines',
                        line=dict(color='purple', width=2),
                        name=f'Offset ({magnitude:.1f}mm)',
                        showlegend=first_offset
                    ),
                    row=3, col=2
                )
                first_offset = False
        
        # Update layout
        fig.update_layout(
            height=1800,
            width=2400,
            title_text="Error Detection Analysis",
            showlegend=True,
            hovermode='closest',
            margin=dict(t=100, b=50, l=50, r=50),
            font=dict(size=14),
            title_font=dict(size=24)
        )
        
        # Update axes labels
        for row in range(1, 4):
            for col in range(1, 3):
                fig.update_xaxes(
                    title_text="Time", 
                    title_font=dict(size=16),
                    row=row, col=col
                )
                fig.update_yaxes(
                    title_text="Water Level (mm)", 
                    title_font=dict(size=16),
                    secondary_y=False,
                    row=row, col=col
                )
        
        # Show the figure
        fig.show_dash(mode='inline')
        
        # Print summary statistics
        print("\nError Detection Summary:")
        print(f"Linear interpolation gaps: {sum(gaps)} found")
        print(f"Noisy periods: {sum(noisy_idx)} measurements")
        print(f"Sustained drops: {sum(sustained_drops)} detected")
        print(f"Point anomalies: {len(anomaly_segments)} segments")
        print(f"Frozen values: {sum(frozen_mask)} measurements")
        print(f"Offsets: {len(offsets)} detected")

    def print_data_characteristics(self):
        """Print key characteristics of the dataset"""
        print("\nData Characteristics:")
        print(f"Time range: {self.df[self.timestamp_column].min()} to {self.df[self.timestamp_column].max()}")
        print(f"Total measurements: {len(self.df)}")
        print(f"Sampling frequency: {self.df['time_diff'].median():.2f} hours")
        
        # Value statistics
        print("\nWater Level Statistics:")
        print(f"Mean: {self.df[self.value_column].mean():.2f}")
        print(f"Median: {self.df[self.value_column].median():.2f}")
        print(f"Std Dev: {self.df[self.value_column].std():.2f}")
        print(f"Min: {self.df[self.value_column].min():.2f}")
        print(f"Max: {self.df[self.value_column].max():.2f}")
        
        # Rate of change statistics
        value_diff = self.df[self.value_column].diff()
        print("\nRate of Change Statistics (per measurement):")
        print(f"Mean change: {value_diff.mean():.2f}")
        print(f"Median change: {value_diff.median():.2f}")
        print(f"Std Dev of change: {value_diff.std():.2f}")
        print(f"Max increase: {value_diff.max():.2f}")
        print(f"Max decrease: {value_diff.min():.2f}")

    def plot_data_overview(self):
        """Plot raw data, edited data, VINGE data, and rainfall."""
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Create a FigureResampler object
        fig = FigureResampler(fig)
        
        # First add VINGE data (in background)
        if hasattr(self, 'vinge_df') and self.vinge_df is not None:
            fig.add_trace(
                go.Scattergl(
                    name='VINGE Data',
                    mode='markers',  # Changed from line to markers
                    marker=dict(
                        color='orange',
                        size=4,
                        symbol='circle'
                    ),
                    showlegend=True,
                    hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
                ),
                hf_x=self.vinge_df['Date'],
                hf_y=self.vinge_df['W.L [cm]'],
                secondary_y=False
            )
        
        # Then add edited data
        if self.edt_df is not None:
            fig.add_trace(
                go.Scattergl(
                    name='Edited Data',
                    line=dict(color='red', width=1, dash='dash'),
                    showlegend=True,
                    hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
                ),
                hf_x=self.edt_df[self.timestamp_column],
                hf_y=self.edt_df[self.value_column],
                secondary_y=False
            )
        
        # Add original data
        fig.add_trace(
            go.Scattergl(
                name='Original Data',
                showlegend=True,
                line=dict(color='black', width=1),
                hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
            ),
            hf_x=self.df[self.timestamp_column],
            hf_y=self.df[self.value_column],
            secondary_y=False
        )
        
        # Add rainfall data after the resampled traces
        if self.rain_df is not None:
            # Calculate the y-axis range for the water level data
            y_min = self.df[self.value_column].min()
            y_max = self.df[self.value_column].max()
            y_range = y_max - y_min
            
            # Place bars starting from the top of the plot
            bar_base = y_max + (y_range * 0.05)
            
            # Get max rainfall value
            max_rainfall = self.rain_df['precipitation (mm)'].max()
            
            # Scale the bars to extend about halfway down
            scaled_rainfall = -(self.rain_df['precipitation (mm)'])  # No scaling factor, just negate
            
            fig.add_trace(
                go.Bar(
                    name='Daily Rainfall',
                    x=self.rain_df['datetime'],
                    y=scaled_rainfall,  # Use negated values
                    marker_color='blue',
                    opacity=0.3,
                    width=24*60*60*1000,  # Set bar width to 1 day in milliseconds
                    showlegend=True,
                    base=bar_base,
                    hovertemplate='Date: %{x}<br>Daily Rainfall: %{customdata:.1f} mm<extra></extra>',
                    customdata=self.rain_df['precipitation (mm)']
                ),
                secondary_y=True  # Changed to use secondary y-axis
            )
            
            # Update primary y-axis for water level data
            fig.update_yaxes(
                title_text="Water Level (mm)",
                title_font=dict(size=16),
                secondary_y=False,
                range=[y_min - (y_range * 0.05), bar_base + (y_range * 0.05)]
            )
            
            # Update secondary y-axis for rainfall with doubled range
            fig.update_yaxes(
                title_text="Daily Rainfall (mm)",
                secondary_y=True,
                range=[max_rainfall * 2, 0],  # Double the range, reversed scale
                overlaying="y"
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1600,
            title_text="Water Level and Rainfall Data Overview",
            showlegend=True,
            hovermode='closest',
            margin=dict(t=100, b=50, l=50, r=50),
            font=dict(size=14),
            title_font=dict(size=24)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", title_font=dict(size=16))
        fig.update_yaxes(
            title_text="Water Level (mm)", 
            title_font=dict(size=16),
            secondary_y=False
        )
        
        # Show the figure
        fig.show_dash(mode='inline')

def get_data_path():
    """Get the path to the data directory."""
    return Path(r"C:\Users\olive\OneDrive\GitHub\MasterThesis\Sample data")

def load_vst_file(file_path):
    """Load a VST file with multiple encoding attempts."""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, 
                           encoding=encoding,
                           delimiter=';',
                           decimal=',',
                           skiprows=3,
                           names=['Date', 'Value'])
            
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return None

def load_edt_file(file_path):
    """Load a VST_EDT file with multiple encoding attempts."""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, 
                           encoding=encoding,
                           delimiter=';',
                           decimal=',',
                           skiprows=3,
                           names=['Date', 'Value'])
            
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return None

def load_vinge_file(file_path):
    """Load and process a VINGE file."""
    try:
        # First try reading with all columns
        df = pd.read_csv(file_path,
                        delimiter='\t',
                        encoding='latin1',
                        decimal=',',
                        quotechar='"',
                        on_bad_lines='skip')
        
        # If that fails, try reading only the needed columns
        if df is None or df.empty:
            df = pd.read_csv(file_path,
                           delimiter='\t',
                           encoding='latin1',
                           decimal=',',
                           quotechar='"',
                           usecols=['Date', 'W.L [cm]'],
                           on_bad_lines='skip')

        # Try multiple date formats
        date_formats = [
            '%d.%m.%Y %H:%M',  # Standard format
            '%Y-%m-%d %H:%M:%S',  # ISO format
            '%d-%m-%Y %H:%M',  # Alternative format
            '%d/%m/%Y %H:%M'   # Another common format
        ]

        for date_format in date_formats:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format=date_format)
                break  # If successful, exit the loop
            except:
                continue
        
        # If none of the specific formats worked, let pandas try to guess
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                print(f"Error: Failed to parse dates in {file_path}")
                return None
        
        # Convert water level to mm and filter years
        df['W.L [cm]'] = pd.to_numeric(df['W.L [cm]'], errors='coerce') * 10  # Convert to mm
        df = df[df['Date'].dt.year >= 1990]
        
        # Drop any rows with NaN values
        df = df.dropna(subset=['Date', 'W.L [cm]'])
        
        if len(df) == 0:
            print(f"Warning: No valid data found in {file_path}")
            return None
            
        return df
    except Exception as e:
        print(f"Error loading VINGE file {file_path}: {str(e)}")
        return None

def load_all_folders(data_dir, folders):
    """Load raw, edited, and VINGE data from all specified folders."""
    all_data = {}
    
    for folder in folders:
        folder_path = data_dir / folder
        if not folder_path.exists():
            print(f"Folder not found: {folder}")
            continue
            
        raw_path = folder_path / "VST_RAW.txt"
        edt_path = folder_path / "VST_EDT.txt"
        vinge_path = folder_path / "VINGE.txt"
        
        if not raw_path.exists():
            print(f"VST_RAW.txt not found in {folder}")
            continue
            
        raw_df = load_vst_file(raw_path)
        edt_df = load_edt_file(edt_path) if edt_path.exists() else None
        vinge_df = load_vinge_file(vinge_path) if vinge_path.exists() else None
        
        if raw_df is not None:
            all_data[folder] = {
                'raw': raw_df,
                'edt': edt_df,
                'vinge': vinge_df
            }
    
    return all_data

def load_rainfall_data(data_dir, station_id):
    """Load rainfall data for a specific station.
    
    Args:
        data_dir (str/Path): Base directory path
        station_id (int): Rain station ID
        
    Returns:
        pd.DataFrame: DataFrame with datetime and rainfall values, or None if file not found
    """
    # Convert data_dir to Path if it isn't already
    data_dir = Path(data_dir)
    
    # Construct path to rain data directory (assuming it's in a 'data' folder next to the base directory)
    rain_data_dir = data_dir.parent / 'data'
    
    # Format station ID with leading zeros
    station_id_padded = f"{int(station_id):05d}"
    rain_file = rain_data_dir / f'RainData_{station_id_padded}.csv'
    
    if rain_file.exists():
        try:
            df = pd.read_csv(rain_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            return df
        except Exception as e:
            print(f"Error loading rainfall data for station {station_id}: {str(e)}")
            return None
    else:
        print(f"Rainfall data file not found for station {station_id}: {rain_file}")
        return None

def main():
    """Main function to analyze errors in the data."""
    folders = ['21006845', '21006846', '21006847']
    data_dir = get_data_path()
    date_range = ('2000-08-14', '2025-09-6')
    
    # Load rain station mapping
    rain_stations = pd.read_csv('data/closest_rain_stations.csv')
    
    all_data = load_all_folders(data_dir, folders)
    
    for folder, data in all_data.items():
        print(f"\nAnalyzing dataset: {folder}")
        
        raw_df = data['raw']
        edt_df = data['edt']
        vinge_df = data['vinge']
        
        # Get corresponding rain station and data
        rain_station = rain_stations[rain_stations['Station_of_Interest'] == int(folder)]['Closest_Rain_Station'].iloc[0]
        rain_data = load_rainfall_data(data_dir, rain_station)
        
        mask = (raw_df['Date'] >= date_range[0]) & (raw_df['Date'] <= date_range[1])
        raw_df_filtered = raw_df[mask].copy()
        
        if edt_df is not None:
            edt_mask = (edt_df['Date'] >= date_range[0]) & (edt_df['Date'] <= date_range[1])
            edt_df_filtered = edt_df[edt_mask].copy()
        else:
            edt_df_filtered = None
            
        if rain_data is not None:
            rain_mask = (rain_data['datetime'] >= date_range[0]) & (rain_data['datetime'] <= date_range[1])
            rain_data_filtered = rain_data[rain_mask].copy()
        else:
            rain_data_filtered = None
        
        if vinge_df is not None:
            vinge_mask = (vinge_df['Date'] >= date_range[0]) & (vinge_df['Date'] <= date_range[1])
            vinge_df_filtered = vinge_df[vinge_mask].copy()
        else:
            vinge_df_filtered = None
        
        if len(raw_df_filtered) == 0:
            print(f"No data found in the specified date range for {folder}")
            continue
            
        print(f"\nAnalyzing data from {date_range[0]} to {date_range[1]}")
        print(f"Number of measurements: {len(raw_df_filtered)} (out of {len(raw_df)} total)")
        
        analyzer = ErrorAnalyzer(raw_df_filtered, edt_df_filtered, rain_data_filtered)
        analyzer.vinge_df = vinge_df_filtered  # Add VINGE data to analyzer
        #analyzer.plot_all_errors()
        analyzer.plot_data_overview()

if __name__ == "__main__":
    main()
