"""
Module for generating synthetic errors in time series data.
Contains functions for injecting various types of errors into clean data segments.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

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
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize error generator with configuration.
        
        Args:
            config: Dictionary of error parameters (if None, uses default from config.py)
        """
        import sys
        from pathlib import Path
        
        # Add project root to path
        project_root = Path(__file__).parents[1]  # Go up two levels from current file
        sys.path.append(str(project_root))
        
        from config import SYNTHETIC_ERROR_PARAMS
        self.config = config or SYNTHETIC_ERROR_PARAMS
        self.error_periods: List[ErrorPeriod] = []
        self.used_indices = set()  # Track all used indices
    
    def _is_period_available(self, start_idx: int, end_idx: int) -> bool:
        """
        Check if a period is available for error injection.
        
        Args:
            start_idx: Start index of proposed period
            end_idx: End index of proposed period
            
        Returns:
            bool: True if period is available, False if there's overlap
        """
        # Check if any index in the range is already used
        period_indices = set(range(start_idx, end_idx))
        return not bool(period_indices & self.used_indices)
    
    def _mark_period_used(self, start_idx: int, end_idx: int):
        """Mark a period as used to prevent overlaps."""
        self.used_indices.update(range(start_idx, end_idx))
    
    def inject_spike_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject spike errors into the time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with injected spike errors
        """
        modified_data = data.copy()
        
        # Get spike parameters from config
        spike_config = self.config['spike']
        frequency = spike_config['frequency']
        mag_range = spike_config['magnitude_range']
        recovery_time = spike_config['recovery_time']
        negative_positiv_ratio = spike_config['negative_positiv_ratio']
        
        # Calculate number of spikes to inject
        n_spikes = int(len(data) * frequency)
        
        # Try to inject spikes until we hit the target or run out of space
        successful_injections = 0
        max_attempts = n_spikes * 10  # Limit attempts to prevent infinite loops
        attempts = 0
        
        while successful_injections < n_spikes and attempts < max_attempts:
            # Randomly select an injection point
            idx = np.random.randint(recovery_time, len(data) - recovery_time)
            end_idx = idx + recovery_time
            
            # Check if period is available
            if self._is_period_available(idx, end_idx):
                # Get the current value and generate spike
                current_value = float(modified_data.iloc[idx].values[0])
                magnitude = current_value * np.random.uniform(*mag_range)
                direction = np.random.choice([-1, 1], p=[negative_positiv_ratio, 1-negative_positiv_ratio])
                spike_value = current_value + (direction * magnitude)
                
                # Ensure spike value is within physical limits
                spike_value = np.clip(
                    spike_value, 
                    self.config.get('PHYSICAL_LIMITS', {}).get('min_value', 0),
                    self.config.get('PHYSICAL_LIMITS', {}).get('max_value', 3000)
                )
                
                # Create and apply recovery pattern
                recovery_pattern = np.linspace(spike_value, current_value, recovery_time + 1)[:-1]
                original_values = modified_data.iloc[idx:idx + recovery_time].copy()
                modified_data.iloc[idx:idx + recovery_time] = recovery_pattern
                
                # Record error period
                self.error_periods.append(
                    ErrorPeriod(
                        start_time=modified_data.index[idx],
                        end_time=modified_data.index[idx + recovery_time - 1],
                        error_type='spike',
                        original_values=original_values.values,
                        modified_values=recovery_pattern,
                        parameters={
                            'magnitude': magnitude,
                            'direction': direction,
                            'recovery_time': recovery_time
                        }
                    )
                )
                
                # Mark period as used
                self._mark_period_used(idx, end_idx)
                successful_injections += 1
            
            attempts += 1
        
        return modified_data

    def inject_flatline_errors(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Inject flatline errors into the time series data.
        Creates periods where the value stays exactly constant (horizontal line).
        """
        modified_data = data.copy()
        flatline_periods = []
        
        # Get flatline parameters from config
        flatline_config = self.config['flatline']
        frequency = flatline_config['frequency']
        duration_range = flatline_config['duration_range']
        
        # Calculate number of flatlines to inject
        n_flatlines = int(len(data) * frequency)
        
        successful_injections = 0
        max_attempts = n_flatlines * 10
        attempts = 0
        
        while successful_injections < n_flatlines and attempts < max_attempts:
            idx = np.random.randint(0, len(data) - duration_range[1])
            duration = np.random.randint(*duration_range)
            end_idx = min(idx + duration, len(modified_data))
            
            if self._is_period_available(idx, end_idx):
                # Store original values
                original_values = modified_data.iloc[idx:end_idx].copy()
                
                # Simply repeat the first value for the entire duration
                flatline_value = float(modified_data.iloc[idx].values[0])
                modified_data.iloc[idx:end_idx] = flatline_value
                
                # Store period for plotting
                flatline_periods.append((modified_data.index[idx], modified_data.index[end_idx-1]))
                
                # Record error period
                self.error_periods.append(
                    ErrorPeriod(
                        start_time=modified_data.index[idx],
                        end_time=modified_data.index[end_idx - 1],
                        error_type='flatline',
                        original_values=original_values.values.flatten(),
                        modified_values=np.full(end_idx - idx, flatline_value),
                        parameters={
                            'duration': duration,
                            'flatline_value': flatline_value
                        }
                    )
                )
                
                self._mark_period_used(idx, end_idx)
                successful_injections += 1
            
            attempts += 1
        
        return modified_data, flatline_periods

    def inject_drift_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject gradual drift errors into the time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with injected drift errors
        """
        modified_data = data.copy()
        
        # Get drift parameters from config
        drift_config = self.config['drift']
        frequency = drift_config['frequency']
        duration_range = drift_config['duration_range']  # in hours
        drift_magnitude = drift_config['magnitude_range']  # maximum deviation at end of drift
        negative_positive_ratio = drift_config['negative_positive_ratio']
        
        # Calculate number of drifts to inject
        n_drifts = int(len(data) * frequency)
        
        successful_injections = 0
        max_attempts = n_drifts * 10
        attempts = 0
        
        while successful_injections < n_drifts and attempts < max_attempts:
            # Randomly select an injection point
            idx = np.random.randint(0, len(data) - duration_range[1])
            duration = np.random.randint(*duration_range)
            end_idx = min(idx + duration, len(modified_data))
            
            if self._is_period_available(idx, end_idx):
                # Store original values
                original_values = modified_data.iloc[idx:end_idx].copy()
                
                # Determine drift direction
                direction = np.random.choice([-1, 1], p=[negative_positive_ratio, 1-negative_positive_ratio])
                
                # Generate drift magnitude
                max_drift = direction * np.random.uniform(*drift_magnitude)
                
                # Create drift pattern (linear or exponential)
                if np.random.random() < 0.5:  # 50% chance of linear vs exponential
                    # Linear drift
                    drift_pattern = np.linspace(0, max_drift, end_idx - idx)
                else:
                    # Exponential drift
                    drift_pattern = max_drift * (np.exp(np.linspace(0, 1, end_idx - idx)) - 1) / (np.e - 1)
                
                # Apply drift to the data
                modified_values = original_values.values.flatten() + drift_pattern
                
                # Ensure values stay within physical limits
                modified_values = np.clip(
                    modified_values,
                    self.config.get('PHYSICAL_LIMITS', {}).get('min_value', 0),
                    self.config.get('PHYSICAL_LIMITS', {}).get('max_value', 3000)
                )
                
                # Apply modified values
                modified_data.iloc[idx:end_idx] = modified_values.reshape(-1, 1)
                
                # Record error period
                self.error_periods.append(
                    ErrorPeriod(
                        start_time=modified_data.index[idx],
                        end_time=modified_data.index[end_idx - 1],
                        error_type='drift',
                        original_values=original_values.values.flatten(),
                        modified_values=modified_values,
                        parameters={
                            'duration': duration,
                            'max_drift': max_drift,
                            'drift_type': 'linear' if np.random.random() < 0.5 else 'exponential'
                        }
                    )
                )
                
                # Mark period as used
                self._mark_period_used(idx, end_idx)
                successful_injections += 1
            
            attempts += 1
            
        return modified_data

    def inject_offset_errors(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Inject sudden offset errors into the time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            Tuple containing:
            - Modified data with injected errors
            - List of tuples representing offset periods
        """
        modified_data = data.copy()
        offset_periods = []
        
        # Get offset parameters from config
        offset_config = self.config['offset']
        frequency = offset_config['frequency']
        mag_range = offset_config['magnitude_range']
        min_duration = offset_config['min_duration']
        negative_positiv_ratio = offset_config['negative_positiv_ratio']
        duration_multiplier_range = offset_config['max_duration_multiplier']
        magnitude_multiplier_range = offset_config['magnitude_multiplier']
        
        # Calculate number of offsets to inject
        n_offsets = int(len(data) * frequency)
        
        # Randomly select injection points, ensuring they don't overlap
        possible_indices = np.arange(0, len(data) - min_duration)
        offset_indices = np.random.choice(possible_indices, size=n_offsets, replace=False)
        
        for idx in offset_indices:
            # Calculate local statistics for context-aware magnitude
            window = slice(max(0, idx-24), min(len(data), idx+24))
            local_std = modified_data.iloc[window].std().values[0]
            
            # Generate offset magnitude with local scaling
            base_magnitude = np.random.uniform(*mag_range)
            local_multiplier = np.random.uniform(*magnitude_multiplier_range)
            magnitude = base_magnitude * local_multiplier * (1 + local_std/100)
            
            # Determine direction with configured ratio
            direction = np.random.choice([-1, 1], p=[negative_positiv_ratio, 1-negative_positiv_ratio])
            offset = direction * magnitude
            
            # Determine variable duration
            max_multiplier = np.random.uniform(*duration_multiplier_range)
            duration = np.random.randint(min_duration, int(min_duration * max_multiplier))
            end_idx = min(idx + duration, len(modified_data))
            
            # Store original values
            original_values = modified_data.iloc[idx:end_idx].copy()
            
            # Create offset values (flatten the array to 1D)
            offset_values = original_values.values.flatten() + offset
            
            # Create the offset series
            offset_series = pd.Series(
                index=modified_data.index[idx:end_idx],
                data=offset_values
            )
            
            # Add vertical transition points
            offset_series.iloc[0] = original_values.iloc[0].values[0]  # Start at original value
            offset_series.iloc[-1] = original_values.iloc[-1].values[0]  # End at original value
            
            # Apply the offset
            modified_data.iloc[idx:end_idx] = offset_series.values.reshape(-1, 1)
            
            # Store period for plotting
            offset_periods.append((modified_data.index[idx], modified_data.index[end_idx-1]))
            
            # Record error period
            self.error_periods.append(
                ErrorPeriod(
                    start_time=modified_data.index[idx],
                    end_time=modified_data.index[end_idx - 1],
                    error_type='offset',
                    original_values=original_values.values.flatten(),
                    modified_values=offset_series.values,
                    parameters={
                        'magnitude': magnitude,
                        'direction': direction,
                        'duration': duration,
                        'local_multiplier': local_multiplier
                    }
                )
            )
            self._mark_period_used(idx, end_idx)
        
        return modified_data, offset_periods

    def inject_noise_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject periods of excessive noise.
        Strategy:
        1. Select periods for increased noise
        2. Add random variations based on local statistics
        3. Maintain physical constraints
        4. Record noise characteristics
        """
        modified_data = data.copy()
        
        # Get noise parameters from config
        noise_config = self.config['noise']
        frequency = noise_config['frequency']
        duration_range = noise_config['duration_range']  # hours
        intensity_range = noise_config['intensity_range']  # multiplier of normal noise level
        
        # Calculate number of noise periods to inject
        n_noise_periods = int(len(data) * frequency)
        
        # Randomly select injection points, ensuring they don't overlap
        possible_indices = np.arange(0, len(data) - duration_range[1])  # Use max duration
        noise_indices = np.random.choice(possible_indices, size=n_noise_periods, replace=False)
        
        for idx in noise_indices:
            # Determine duration of noise period
            duration = np.random.randint(duration_range[0], duration_range[1])
            end_idx = min(idx + duration, len(modified_data))
            
            if self._is_period_available(idx, end_idx):
                # Store original values
                original_values = modified_data.iloc[idx:end_idx].copy()
                
                # Add random noise to the middle section
                if end_idx - idx > 2:  # If there's room for a middle section
                    noise_level = np.random.uniform(*intensity_range)
                    local_std = modified_data.iloc[idx:end_idx]['Value'].std()
                    noise = np.random.normal(0, local_std * noise_level, size=end_idx-idx-2)
                    modified_data.iloc[idx+1:end_idx-1, 0] += noise
                
                # Create transition at end (sudden jump back)
                if end_idx < len(modified_data):
                    modified_data.iloc[end_idx-1] = float(modified_data.iloc[end_idx].values[0])
                
                # Record error period
                self.error_periods.append(
                    ErrorPeriod(
                        start_time=modified_data.index[idx],
                        end_time=modified_data.index[end_idx - 1],
                        error_type='noise',
                        original_values=original_values.values,
                        modified_values=modified_data.iloc[idx:end_idx].values,
                        parameters={
                            'noise_level': noise_level,
                            'duration': duration
                        }
                    )
                )
                
                self._mark_period_used(idx, end_idx)

        return modified_data

    def inject_baseline_shift_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject sudden baseline shift errors that permanently change the base level.
        Creates abrupt, permanent changes in the baseline level, similar to sensor recalibration
        or physical changes in measurement conditions.
        """
        modified_data = data.copy()
        
        # Get baseline shift parameters from config
        shift_config = self.config['baseline_shift']
        frequency = shift_config['frequency']
        magnitude_range = shift_config['magnitude_range']
        negative_positive_ratio = shift_config['negative_positive_ratio']
        
        # Calculate number of shifts to inject
        n_shifts = int(len(data) * frequency)
        
        successful_injections = 0
        max_attempts = n_shifts * 10
        attempts = 0
        
        while successful_injections < n_shifts and attempts < max_attempts:
            # Randomly select an injection point
            idx = np.random.randint(0, len(data) - 2)  # Leave room for the shift
            
            # We only need 2 points for the shift (almost vertical)
            end_idx = idx + 2
            
            if self._is_period_available(idx, end_idx):
                # Store original values
                original_values = modified_data.iloc[idx:end_idx].copy()
                
                # Determine shift direction and magnitude
                direction = np.random.choice([-1, 1], p=[negative_positive_ratio, 1-negative_positive_ratio])
                magnitude = direction * np.random.uniform(*magnitude_range)
                
                # Apply the shift (abrupt change)
                modified_data.iloc[idx+1:] += magnitude  # Everything after the shift point
                
                # Record error period
                self.error_periods.append(
                    ErrorPeriod(
                        start_time=modified_data.index[idx],
                        end_time=modified_data.index[idx + 1],
                        error_type='baseline_shift',
                        original_values=original_values.values.flatten(),
                        modified_values=modified_data.iloc[idx:idx+2].values.flatten(),
                        parameters={
                            'magnitude': magnitude
                        }
                    )
                )
                
                # Mark shift point as used
                self._mark_period_used(idx, end_idx)
                successful_injections += 1
            
            attempts += 1
        
        return modified_data

    def inject_all_errors(self, data: pd.DataFrame, 
                         error_types: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject all types of errors into the data.
        
        Args:
            data: DataFrame with time series data
            error_types: List of error types to inject. If None, injects all types.
                        Options: ['spike', 'flatline', 'drift', 'offset', 'baseline_shift', 'noise']
        
        Returns:
            Tuple of (modified_data, ground_truth)
        """
        modified_data = data.copy()
        all_error_types = ['spike', 'flatline', 'drift', 'offset', 'baseline_shift', 'noise']
        error_types = error_types or all_error_types
        
        print(f"Injecting error types: {error_types}")
        
        if 'spike' in error_types:
            modified_data = self.inject_spike_errors(modified_data)
        
        if 'flatline' in error_types:
            modified_data, flatline_periods = self.inject_flatline_errors(modified_data)
        
        if 'drift' in error_types:
            modified_data = self.inject_drift_errors(modified_data)
        
        if 'baseline_shift' in error_types:
            modified_data = self.inject_baseline_shift_errors(modified_data)
        
        if 'offset' in error_types:
            modified_data, offset_periods = self.inject_offset_errors(modified_data)
        
        if 'noise' in error_types:
            modified_data = self.inject_noise_errors(modified_data)
        
        # Create ground truth labels (for future use)
        ground_truth = pd.DataFrame(index=data.index)
        ground_truth['error'] = False
        
        for period in self.error_periods:
            mask = (ground_truth.index >= period.start_time) & (ground_truth.index <= period.end_time)
            ground_truth.loc[mask, 'error'] = True
            ground_truth.loc[mask, 'error_type'] = period.error_type
        
        return modified_data, ground_truth

#Testing the synthetic error generator
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import data_loading
    sys.path.append(str(Path(__file__).parents[2]))
    from data_utils.data_loading import load_vst_file
    
    # Load and prepare data
    data_path = Path(r"C:\Users\olive\OneDrive\GitHub\MasterThesis\project_code\data_utils\Sample data\21006845\VST_RAW.txt")
    sample_data = load_vst_file(data_path)
    
    # Filter for desired time period
    mask = (sample_data['Date'] >= '2023-01-01') & (sample_data['Date'] < '2024-01-01')
    sample_data = sample_data.loc[mask].copy()
    
    # Set Date as index and rename Value column
    sample_data.set_index('Date', inplace=True)
    sample_data = sample_data.rename(columns={'Value': 'water_level'})
    
    # Test error injection
    generator = SyntheticErrorGenerator()
    data_with_spikes = generator.inject_spike_errors(sample_data)
    data_with_flatlines, flatline_periods = generator.inject_flatline_errors(data_with_spikes)
    data_with_drift = generator.inject_drift_errors(data_with_flatlines)
    data_with_baseline = generator.inject_baseline_shift_errors(data_with_drift)
    modified_data, offset_periods = generator.inject_offset_errors(data_with_baseline)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Water Level Time Series with Synthetic Errors', fontsize=14)
    
    # After creating the subplots but before plotting data
    # Get the overall min and max values
    y_min = min(sample_data.min().min(), modified_data.min().min())
    y_max = max(sample_data.max().max(), modified_data.max().max())
    
    # Add some padding (e.g., 5% of the range)
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding
    
    # Set the same y-axis limits for both plots
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    # Define colors for different error types
    ERROR_COLORS = {
        'base': '#1f77b4',      # Default blue
        'spike': '#ff7f0e',     # Orange
        'baseline_shift': '#2ca02c',  # Green
        'offset': '#d62728',    # Red
        'drift': '#ff7f0e',     # Yellow
        'flatline': '#8c564b'  # Brown
    }
    
    # Plot original data in first subplot
    ax1.plot(sample_data, label='Original Data', color=ERROR_COLORS['base'], alpha=0.7)
    
    # Plot base data first with lower zorder so it is drawn behind the errors
    ax2.plot(modified_data, color=ERROR_COLORS['base'], alpha=0.7, label='Base Data', zorder=1)
    
    # Plot spikes with higher zorder and optional markers
    spike_line = None  # Store the first spike line for legend
    for period in generator.error_periods:
        if period.error_type == 'spike':
            # Create a temporary series for the spike values
            spike_series = pd.Series(
                index=modified_data.loc[period.start_time:period.end_time].index,
                data=period.modified_values
            )
            line = ax2.plot(
                spike_series, 
                color=ERROR_COLORS['spike'], 
                alpha=1.0, 
                linewidth=2,
                marker='o',      
                zorder=3,        
                label='_nolegend_'  # Don't add to legend automatically
            )[0]  # Get the line object
            if spike_line is None:
                spike_line = line
    
    # Plot flatline periods with vertical indicators
    flatline_line = None  # Store the first flatline for legend
    for start, end in flatline_periods:
        mask = (modified_data.index >= start) & (modified_data.index <= end)
        line = ax2.plot(
            modified_data[mask], 
            color=ERROR_COLORS['flatline'], 
            alpha=0.9, 
            linewidth=2,
            label='_nolegend_'
        )[0]  # Get the line object
        if flatline_line is None:
            flatline_line = line
        # Add vertical line indicator in the middle
        middle_time = start + (end - start) / 2
        ax2.axvline(x=middle_time, color=ERROR_COLORS['flatline'], alpha=0.3, linewidth=8)
    
    # Plot offset periods with vertical indicators
    offset_line = None  # Store the first offset for legend
    for start, end in offset_periods:
        mask = (modified_data.index >= start) & (modified_data.index <= end)
        # Plot the offset
        line = ax2.plot(
            modified_data[mask], 
            color=ERROR_COLORS['offset'], 
            alpha=0.9, 
            linewidth=2,
            label='_nolegend_'
        )[0]  # Get the line object
        if offset_line is None:
            offset_line = line
        # Add vertical line indicator in the middle
        middle_time = start + (end - start) / 2
        ax2.axvline(x=middle_time, color=ERROR_COLORS['offset'], alpha=0.3, linewidth=8)
    
    # Add drift plotting with yellow background
    drift_line = None
    for period in generator.error_periods:
        if period.error_type == 'drift':
            mask = (modified_data.index >= period.start_time) & (modified_data.index <= period.end_time)
            # Add yellow background for drift period
            ax2.axvspan(period.start_time, period.end_time, 
                       color='yellow', alpha=0.2, zorder=1)
            # Plot the drift line
            line = ax2.plot(
                modified_data[mask],
                color=ERROR_COLORS['drift'],
                alpha=0.9,
                linewidth=2,
                label='_nolegend_',
                zorder=3
            )[0]
            if drift_line is None:
                drift_line = line
    
    # Add plotting code for baseline shifts:
    baseline_line = None
    for period in generator.error_periods:
        if period.error_type == 'baseline_shift':
            mask = (modified_data.index >= period.start_time) & (modified_data.index <= period.end_time)
            # Add vertical line to mark the shift
            ax2.axvline(x=period.start_time, 
                       color=ERROR_COLORS['baseline_shift'], 
                       alpha=0.8, 
                       linewidth=2,
                       linestyle='--')
            # Plot the transition period
            line = ax2.plot(
                modified_data[mask],
                color=ERROR_COLORS['baseline_shift'],
                alpha=0.9,
                linewidth=2,
                label='_nolegend_',
                zorder=3
            )[0]
            if baseline_line is None:
                baseline_line = line
    
    ax2.set_ylabel('Water Level')
    ax2.grid(False)
    
    # Update legend to include baseline shifts:
    ax2.legend([
        spike_line,
        flatline_line,
        offset_line,
        drift_line,
        baseline_line
    ], ['Spikes', 'Flatlines', 'Offsets', 'Drifts', 'Baseline Shifts'])
    
    ax2.set_title('Data with Injected Errors')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
