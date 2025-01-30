import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ErrorAnalyzer:
    def __init__(self, raw_df: pd.DataFrame, edt_df: pd.DataFrame = None, 
                 value_column: str = 'Value', timestamp_column: str = 'Date'):
        """
        Initialize the analyzer with raw and edited dataframes.
        
        Args:
            raw_df: DataFrame with raw water level measurements
            edt_df: DataFrame with edited water level measurements (optional)
            value_column: Name of the column containing water level values
            timestamp_column: Name of the column containing timestamps
        """
        self.df = raw_df.copy()
        self.edt_df = edt_df.copy() if edt_df is not None else None
        self.value_column = value_column
        self.timestamp_column = timestamp_column
        
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

    def detect_frozen_values(self, 
                           variation_tolerance: float = 0.01, 
                           window_hours: float = 12):  # 12 hours is good for frozen values
        """Detect frozen values - shorter window as frozen values are very stable."""
        window_size = int(window_hours * self.SAMPLES_PER_HOUR)
        return self.df[self.value_column].diff().abs().rolling(window_size).max() < variation_tolerance

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
        """Plot all error types using the detection functions."""
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 18))
        axes = axes.flatten()  # Flatten the 2D array to 1D for easier indexing
        ax1, ax2, ax3, ax4, ax5, ax6 = axes
        
        # Plot edited data if available
        if self.edt_df is not None:
            for ax in axes:
                ax.plot(self.edt_df[self.timestamp_column], 
                       self.edt_df[self.value_column], 
                       'r--', linewidth=1, alpha=0.7, label='Edited Data')

        # Plot original data on all subplots
        for ax in axes:  # Changed to include all subplots
            ax.plot(self.df[self.timestamp_column], self.df[self.value_column], 
                   'k-', linewidth=1, label='Original Data')

        # 1. Linear interpolation
        gaps = self.detect_gaps()
        if any(gaps):
            legend_plotted = False
            gap_periods = self.df[gaps].index
            for idx in gap_periods:
                if idx+1 < len(self.df):
                    label = 'Linear Interpolation' if not legend_plotted else None
                    ax1.plot([self.df[self.timestamp_column].iloc[idx], 
                             self.df[self.timestamp_column].iloc[idx+1]],
                            [self.df[self.value_column].iloc[idx], 
                             self.df[self.value_column].iloc[idx+1]],
                            'purple', linewidth=2, alpha=0.8, label=label)
                    legend_plotted = True

        # 2. Noise
        noisy_idx = self.detect_noise()
        if any(noisy_idx):
            # Convert boolean mask to ranges of noisy periods
            noisy_ranges = []
            start_idx = None
            for i, is_noisy in enumerate(noisy_idx):
                if is_noisy and start_idx is None:
                    start_idx = i
                elif not is_noisy and start_idx is not None:
                    noisy_ranges.append((start_idx, i))
                    start_idx = None
            # Add final range if series ends while noisy
            if start_idx is not None:
                noisy_ranges.append((start_idx, len(noisy_idx)))
            
            # Plot vertical spans for each noisy period
            for start, end in noisy_ranges:
                ax2.axvspan(self.df[self.timestamp_column].iloc[start],
                           self.df[self.timestamp_column].iloc[end],
                           color='yellow', alpha=0.3, label='Noisy Period' if start == noisy_ranges[0][0] else None)

        # 3. Drops
        sustained_drops = self.detect_drops()
        if any(sustained_drops):
            ax3.fill_between(self.df[self.timestamp_column][sustained_drops],
                            self.df[self.value_column][sustained_drops],
                            self.df[self.value_column][sustained_drops] * 0.9,
                            color='green', alpha=0.4, label='Sustained Drops')

        # 4. Point anomalies
        anomaly_segments = self.detect_point_anomalies()
        legend_added = False
        for start, end in anomaly_segments:
            # Ensure start and end are integers
            start = int(start)
            end = int(end)
            label = 'Point Anomalies' if not legend_added else None
            ax4.plot(self.df[self.timestamp_column].iloc[start:end],
                    self.df[self.value_column].iloc[start:end],
                    'b-', linewidth=2, alpha=0.8, label=label)
            legend_added = True

        # 5. Frozen values
        frozen_mask = self.detect_frozen_values()
        if any(frozen_mask):
            # Convert boolean mask to ranges of frozen periods
            frozen_ranges = []
            start_idx = None
            for i, is_frozen in enumerate(frozen_mask):
                if is_frozen and start_idx is None:
                    start_idx = i
                elif not is_frozen and start_idx is not None:
                    frozen_ranges.append((start_idx, i))
                    start_idx = None
            # Add final range if series ends while frozen
            if start_idx is not None:
                frozen_ranges.append((start_idx, len(frozen_mask)))
            
            # Plot vertical spans for each frozen period
            for start, end in frozen_ranges:
                ax5.axvspan(self.df[self.timestamp_column].iloc[start],
                           self.df[self.timestamp_column].iloc[end],
                           color='gray', alpha=0.3, 
                           label='Frozen Values' if start == frozen_ranges[0][0] else None)

        # 6. Offset Detection (moved from last)
        offsets = self.detect_offsets()
        if offsets:
            for start, end, magnitude in offsets:
                # Plot vertical line at offset point
                ax6.axvline(self.df[self.timestamp_column].iloc[start], 
                           color='purple', linestyle='--', alpha=0.8,
                           label=f'Offset ({magnitude:.1f}mm)' if start == offsets[0][0] else None)
                
                # Highlight the transition period
                ax6.axvspan(self.df[self.timestamp_column].iloc[start],
                           self.df[self.timestamp_column].iloc[end],
                           color='purple', alpha=0.2)

        # Set titles and format
        titles = ['Linear Interpolation Detection', 'Noise Detection', 
                 'Sustained Drop Detection', 'Point Anomaly Detection', 
                 'Frozen Value Detection', 'Offset Detection']  # Updated last title
        
        for ax, title in zip(axes, titles):
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Water Level')
            ax.set_xlabel('Time')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.legend(loc='upper right', ncol=1)

        plt.suptitle('Error Detection Analysis', fontsize=14)
        plt.tight_layout()
        plt.show()

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

def load_all_folders(data_dir, folders):
    """Load both raw and edited data from all specified folders."""
    all_data = {}
    
    for folder in folders:
        folder_path = data_dir / folder
        if not folder_path.exists():
            print(f"Folder not found: {folder}")
            continue
            
        raw_path = folder_path / "VST_RAW.txt"
        edt_path = folder_path / "VST_EDT.txt"
        
        if not raw_path.exists():
            print(f"VST_RAW.txt not found in {folder}")
            continue
            
        raw_df = load_vst_file(raw_path)
        edt_df = load_edt_file(edt_path) if edt_path.exists() else None
        
        if raw_df is not None:
            all_data[folder] = {
                'raw': raw_df,
                'edt': edt_df
            }
    
    return all_data

def main():
    """Main function to analyze errors in the data."""
    folders = ['21006845', '21006846', '21006847']
    data_dir = get_data_path()
    date_range = ('2010-08-14', '2016-09-6')
    
    all_data = load_all_folders(data_dir, folders)
    
    for folder, data in all_data.items():
        print(f"\nAnalyzing dataset: {folder}")
        
        raw_df = data['raw']
        edt_df = data['edt']
        
        mask = (raw_df['Date'] >= date_range[0]) & (raw_df['Date'] <= date_range[1])
        raw_df_filtered = raw_df[mask].copy()
        
        if edt_df is not None:
            edt_mask = (edt_df['Date'] >= date_range[0]) & (edt_df['Date'] <= date_range[1])
            edt_df_filtered = edt_df[edt_mask].copy()
        else:
            edt_df_filtered = None
        
        if len(raw_df_filtered) == 0:
            print(f"No data found in the specified date range for {folder}")
            continue
            
        print(f"\nAnalyzing data from {date_range[0]} to {date_range[1]}")
        print(f"Number of measurements: {len(raw_df_filtered)} (out of {len(raw_df)} total)")
        
        analyzer = ErrorAnalyzer(raw_df_filtered, edt_df_filtered)
        analyzer.plot_all_errors()
        analyzer.print_data_characteristics()

if __name__ == "__main__":
    main()

