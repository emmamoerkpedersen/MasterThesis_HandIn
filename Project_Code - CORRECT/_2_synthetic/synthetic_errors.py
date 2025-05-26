"""
Module for generating synthetic errors in time series data.
Contains functions for injecting various types of errors into clean data segments.

Potential Future Enhancements:
1. Varied Spike Types:
   - Sudden: Very quick, subtle to medium spikes
   - Gradual: Slower recovery spikes
   - Extreme: More obvious spikes
   - Configurable probability distribution for each type

2. Pattern-Based Anomalies:
   - Mini oscillations with subtle amplitudes
   - Temporary trend changes
   - Step patterns with varying durations
   - Complex pattern combinations

3. Enhanced Drift Patterns:
   - Multiple drift types (linear, exponential, logarithmic)
   - Compound drifts (combinations of patterns)
   - Seasonal-aware drift magnitudes

4. Noise-Based Anomalies:
   - Variable noise levels
   - Burst noise patterns
   - Signal-to-noise ratio based anomalies
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
        self.error_periods = []
        self.used_indices = set()  # Track all used indices

    def _calculate_years_in_data(self, time_index):
        """
        Calculate approximately how many years of data are in the time index.
        This is used to scale the total number of errors to inject based on
        the configured count_per_year parameter. The errors are then distributed
        randomly across the entire dataset, not per year.
        
        Args:
            time_index: pandas DatetimeIndex
            
        Returns:
            float: Number of years in the data
        """
        if len(time_index) < 2:
            return 1.0
            
        # Calculate time span in days
        start_date = time_index[0]
        end_date = time_index[-1]
        days = (end_date - start_date).total_seconds() / (24 * 3600)
        
        # Convert to years (approximate)
        return max(1.0, days / 365.25)
    
    def _is_period_available(self, start_idx: int, end_idx: int, buffer: int = 24) -> bool:
        """
        Check if a period is available for error injection.
        Includes a buffer zone around existing errors.
        
        Args:
            start_idx: Start index of proposed period
            end_idx: End index of proposed period
            buffer: Number of indices to keep clear around errors (default: 24 hours)
            
        Returns:
            bool: True if period is available, False if there's overlap
        """
        # Check a wider range that includes the buffer
        buffered_start = max(0, start_idx - buffer)
        buffered_end = end_idx + buffer
        
        # Check if any index in the range (including buffer) is already used
        period_indices = set(range(buffered_start, buffered_end))
        return not bool(period_indices & self.used_indices)
    
    def _mark_period_used(self, start_idx: int, end_idx: int):
        """Mark a period as used to prevent overlaps."""
        self.used_indices.update(range(start_idx, end_idx))

    def inject_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject missing data anomalies by replacing sections with NaN values.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with injected missing data sections
        """
        modified_data = data.copy()
        
        # Get missing data parameters from config
        missing_config = self.config.get('missing_data', {})
        count_per_year = missing_config.get('count_per_year', 0)
        
        # If count_per_year is 0, don't inject any missing data
        if count_per_year == 0:
            print("Missing data injection disabled (count_per_year = 0)")
            return modified_data
            
        min_length = missing_config.get('min_length', 100)
        max_length = missing_config.get('max_length', 500)
        
        # Calculate years in data and total number of missing data periods
        years_in_data = self._calculate_years_in_data(data.index)
        n_missing_periods = round(count_per_year * years_in_data)
        
        # If no periods to inject, return original data
        if n_missing_periods == 0:
            print("No missing data periods to inject")
            return modified_data
            
        print(f"Data spans approximately {years_in_data:.1f} years")
        print(f"Attempting to inject {n_missing_periods} missing data periods...")
        
        successful_injections = 0
        max_attempts = n_missing_periods * 10
        attempts = 0
        
        while successful_injections < n_missing_periods and attempts < max_attempts:
            attempts += 1
            
            try:
                # Randomly determine the length of the missing data segment
                missing_length = np.random.randint(min_length, max_length + 1)
                
                # Select a random starting point, avoiding the very beginning and end
                start_idx = np.random.randint(10, len(data) - missing_length - 10)
                end_idx = start_idx + missing_length
                
                # Check if this period is available (not overlapping with other errors)
                if not self._is_period_available(start_idx, end_idx):
                    continue
                
                # Store original values
                original_values = modified_data.iloc[start_idx:end_idx]['vst_raw'].copy().values
                
                # Replace values with NaN
                modified_data.iloc[start_idx:end_idx, modified_data.columns.get_loc('vst_raw')] = np.nan
                
                # Record the error period
                self.error_periods.append(
                    ErrorPeriod(
                        start_time=modified_data.index[start_idx],
                        end_time=modified_data.index[end_idx - 1],
                        error_type='missing_data',
                        original_values=original_values,
                        modified_values=np.full_like(original_values, np.nan),
                        parameters={'missing_length': missing_length}
                    )
                )
                
                # Mark period as used
                self._mark_period_used(start_idx, end_idx)
                successful_injections += 1
                print(f"Successfully injected missing data period {successful_injections}/{n_missing_periods} - {missing_length} points")
                
            except Exception as e:
                print(f"Failed to inject missing data at index {start_idx}:{end_idx}")
                print(f"Error: {str(e)}")
                continue
        
        return modified_data

    def inject_spike_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject spike errors into the time series data.
        
        Calculates the total number of spikes based on data span (years) and configured 
        count_per_year, then distributes them randomly throughout the entire dataset.
        """
        modified_data = data.copy()
        
        # Get spike parameters from config
        spike_config = self.config['spike']
        count_per_year = spike_config.get('count_per_year', 0)
        
        # If count_per_year is 0, don't inject any spikes
        if count_per_year == 0:
            print("Spike injection disabled (count_per_year = 0)")
            return modified_data
            
        mag_range = spike_config['magnitude_range']
        negative_positiv_ratio = spike_config['negative_positiv_ratio']
        
        # Calculate years in data and total number of spikes
        years_in_data = self._calculate_years_in_data(data.index)
        # Ensure at least 1 spike if count_per_year > 0
        n_spikes = max(1, round(count_per_year * years_in_data))
        
        print(f"Data spans approximately {years_in_data:.1f} years")
        print(f"Attempting to inject {n_spikes} spikes randomly across entire dataset...")
        
        successful_injections = 0
        max_attempts = n_spikes * 10
        attempts = 0
        
        while successful_injections < n_spikes and attempts < max_attempts:
            attempts += 1
            
            try:
                # Select injection point randomly from the entire dataset (avoid edges)
                idx = np.random.randint(1, len(data) - 1)
                
                if not self._is_period_available(idx, idx + 1):
                    continue
                
                # Get current value and values before/after
                current_value = float(data.iloc[idx]['vst_raw'])
                
                # Generate spike magnitude
                magnitude = np.random.uniform(*mag_range)
                
                # Determine spike direction
                direction = np.random.choice([-1, 1], p=[negative_positiv_ratio, 1-negative_positiv_ratio])
                
                # Calculate spike value with physical limits
                spike_value = np.clip(
                    current_value + (direction * magnitude),
                    self.config.get('PHYSICAL_LIMITS', {}).get('min_value', 0),
                    self.config.get('PHYSICAL_LIMITS', {}).get('max_value', 3000)
                )
                
                # Store original value
                original_value = current_value
                
                # Apply the spike (single point)
                modified_data.iloc[idx, modified_data.columns.get_loc('vst_raw')] = spike_value
                
                # Record error period
                self.error_periods.append(
                    ErrorPeriod(
                        start_time=modified_data.index[idx],
                        end_time=modified_data.index[idx],  # Same as start for single point
                        error_type='spike',
                        original_values=np.array([original_value]),
                        modified_values=np.array([spike_value]),
                        parameters={
                            'magnitude': magnitude,
                            'direction': direction
                        }
                    )
                )
                
                # Mark period as used
                self._mark_period_used(idx, idx + 1)
                successful_injections += 1
                #print(f"Successfully injected spike {successful_injections}/{n_spikes}")
                
            except Exception as e:
                print(f"Failed to inject spike at index {idx}")
                print(f"Error: {str(e)}")
                continue
        
        return modified_data

    def inject_flatline_errors(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Inject flatline errors into the time series data.
        Creates periods where the value stays exactly constant (horizontal line).
        
        Calculates the total number of flatlines based on data span (years) and configured
        count_per_year, then distributes them randomly throughout the entire dataset.
        """
        modified_data = data.copy()
        flatline_periods = []
        
        # Get flatline parameters from config
        flatline_config = self.config['flatline']
        count_per_year = flatline_config.get('count_per_year', 0)
        duration_range = flatline_config['duration_range']
        
        # Calculate years in data and total number of flatlines
        years_in_data = self._calculate_years_in_data(data.index)
        n_flatlines = int(count_per_year * years_in_data)
        
        print(f"Attempting to inject {n_flatlines} flatlines randomly across entire dataset...")
        
        successful_injections = 0
        max_attempts = n_flatlines * 10
        attempts = 0
        
        while successful_injections < n_flatlines and attempts < max_attempts:
            # Select random index from entire dataset
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
        
        Calculates the total number of drifts based on data span (years) and configured
        count_per_year, then distributes them randomly throughout the entire dataset.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with injected drift errors
        """
        modified_data = data.copy()
        
        drift_config = self.config['drift']
        count_per_year = drift_config.get('count_per_year', 0)
        
        # Calculate years in data and total number of drifts
        years_in_data = self._calculate_years_in_data(data.index)
        n_drifts = max(1, int(count_per_year * years_in_data))
        
        print(f"Attempting to inject {n_drifts} drifts randomly across entire dataset...")
        
        successful_injections = 0
        max_attempts = n_drifts * 10
        attempts = 0
        
        while successful_injections < n_drifts and attempts < max_attempts:
            # Randomly select an injection point from entire dataset
            idx = np.random.randint(0, len(data) - drift_config['duration_range'][1])
            duration = np.random.randint(*drift_config['duration_range'])
            end_idx = min(idx + duration, len(modified_data))
            
            if self._is_period_available(idx, end_idx):
                # Store original values
                original_values = modified_data.iloc[idx:end_idx].copy()
                
                # Determine drift direction
                direction = np.random.choice([-1, 1], p=[drift_config['negative_positive_ratio'], 1-drift_config['negative_positive_ratio']])
                
                # Generate drift magnitude
                max_drift = direction * np.random.uniform(*drift_config['magnitude_range'])
                
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
        
        Calculates the total number of offsets based on data span (years) and configured
        count_per_year, then distributes them randomly throughout the entire dataset.
        
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
        count_per_year = offset_config.get('count_per_year', 0)
        
        # If count_per_year is 0, don't inject any offsets
        if count_per_year == 0:
            print("Offset injection disabled (count_per_year = 0)")
            return modified_data, offset_periods
            
        duration_range = offset_config['duration_range']
        max_duration = min(
            duration_range[1],
            len(data) // 5  # Limit to 20% of data length
        )
        mag_range = offset_config['magnitude_range']
        negative_positiv_ratio = offset_config['negative_positiv_ratio']
        
        # Calculate years in data and total number of offsets
        years_in_data = self._calculate_years_in_data(data.index)
        # Ensure at least 1 offset if count_per_year > 0
        n_offsets = max(1, round(count_per_year * years_in_data))
        
        print(f"Data spans approximately {years_in_data:.1f} years")
        print(f"Attempting to inject {n_offsets} offset periods randomly across entire dataset...")
        
        # Randomly select injection points from entire dataset
        possible_indices = np.arange(0, len(data) - duration_range[0])
        
        # Try to place offsets while respecting constraints
        successful_injections = 0
        max_attempts = n_offsets * 10
        attempts = 0
        
        while successful_injections < n_offsets and attempts < max_attempts:
            # Select a random index from entire dataset
            idx = np.random.choice(possible_indices)
            attempts += 1
            
            # Generate offset magnitude
            magnitude = np.random.uniform(*mag_range)
            
            # Determine direction with configured ratio
            direction = np.random.choice([-1, 1], p=[negative_positiv_ratio, 1-negative_positiv_ratio])
            offset = direction * magnitude
            
            # Determine variable duration
            duration = np.random.randint(duration_range[0], max_duration)
            end_idx = min(idx + duration, len(modified_data))
            
            # Check if period is available
            if not self._is_period_available(idx, end_idx):
                continue
                
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
                        'duration': duration
                    }
                )
            )
            self._mark_period_used(idx, end_idx)
            successful_injections += 1
        
        return modified_data, offset_periods

    def inject_noise_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject periods of excessive noise.
        
        Calculates the total number of noise periods based on data span (years) and configured
        count_per_year, then distributes them randomly throughout the entire dataset.
        
        Strategy:
        1. Select periods for increased noise
        2. Add random variations based on local statistics
        3. Maintain physical constraints
        4. Record noise characteristics
        """
        modified_data = data.copy()
        
        # Get noise parameters from config
        noise_config = self.config['noise']
        count_per_year = noise_config.get('count_per_year', 0)
        
        # If count_per_year is 0, don't inject any noise
        if count_per_year == 0:
            print("Noise injection disabled (count_per_year = 0)")
            return modified_data
            
        duration_range = noise_config['duration_range']  # hours
        # intensity_range is now used for the noise around the new level, applied to segment_noise_std_abs
        intensity_range = noise_config.get('intensity_range', (1.0, 1.0)) 

        # New parameters for stepped noise
        num_sub_segments_range = noise_config.get('num_sub_segments_range', (2, 4))
        segment_level_offset_range_abs = noise_config.get('segment_level_offset_range_abs', (20, 50))
        segment_noise_std_abs = noise_config.get('segment_noise_std_abs', 5)
        
        # Calculate years in data and total number of noise periods
        years_in_data = self._calculate_years_in_data(data.index)
        # Ensure at least 1 noise period if count_per_year > 0
        n_noise_periods = max(1, round(count_per_year * years_in_data))
        
        print(f"Data spans approximately {years_in_data:.1f} years")
        print(f"Attempting to inject {n_noise_periods} noise periods randomly across entire dataset...")
        
        # Try to place noise periods while respecting constraints
        successful_injections = 0
        max_attempts = n_noise_periods * 10
        attempts = 0
        
        while successful_injections < n_noise_periods and attempts < max_attempts:
            # Select a random index from entire dataset
            idx = np.random.randint(0, len(data) - duration_range[1])
            attempts += 1
            
            # Determine duration of noise period
            duration = np.random.randint(duration_range[0], duration_range[1])
            end_idx = min(idx + duration, len(modified_data))
            
            if self._is_period_available(idx, end_idx):
                # Store original values for the entire noise period for logging
                original_values_for_log = modified_data.iloc[idx:end_idx].copy()
                
                num_sub_segments = np.random.randint(num_sub_segments_range[0], num_sub_segments_range[1] + 1)
                segment_len = (end_idx - idx) // num_sub_segments
                
                current_actual_idx = idx
                period_parameters = {'segments': []}

                for i_segment in range(num_sub_segments):
                    seg_start_idx = current_actual_idx
                    seg_end_idx = current_actual_idx + segment_len if i_segment < num_sub_segments - 1 else end_idx

                    if seg_start_idx >= seg_end_idx: # Should not happen if segment_len > 0
                        continue

                    # Determine the new base level for this segment
                    # Use the original data's value at the start of the segment as a reference point
                    reference_value_for_level = data.iloc[seg_start_idx]['vst_raw'] 
                    if pd.isna(reference_value_for_level):
                         # Fallback if original data is NaN: use mean of local window or overall mean
                        local_window_mean = data.iloc[max(0, seg_start_idx-24):min(len(data), seg_start_idx+24)]['vst_raw'].mean()
                        reference_value_for_level = local_window_mean if pd.notna(local_window_mean) else data['vst_raw'].mean()
                    
                    level_offset = np.random.uniform(segment_level_offset_range_abs[0],
                                                   segment_level_offset_range_abs[1]) * np.random.choice([-1, 1])
                    new_segment_base_level = reference_value_for_level + level_offset

                    # Add noise around this new base level
                    current_intensity = np.random.uniform(intensity_range[0], intensity_range[1])
                    noise_for_segment = np.random.normal(0, 
                                                         segment_noise_std_abs * current_intensity, 
                                                         size=(seg_end_idx - seg_start_idx))
                    
                    modified_data.iloc[seg_start_idx:seg_end_idx, 0] = new_segment_base_level + noise_for_segment
                    
                    period_parameters['segments'].append({
                        'start_idx': seg_start_idx,
                        'end_idx': seg_end_idx -1,
                        'base_level': new_segment_base_level,
                        'level_offset': level_offset,
                        'noise_intensity_factor': current_intensity
                    })
                    current_actual_idx = seg_end_idx

                # Apply physical limits (optional, but good practice)
                # This part can be adapted from how physical limits are handled elsewhere
                # For now, simple clipping as an example:
                min_limit = self.config.get('PHYSICAL_LIMITS', {}).get('min_value', 0)
                max_limit = self.config.get('PHYSICAL_LIMITS', {}).get('max_value', 3000)
                modified_data.iloc[idx:end_idx, 0] = np.clip(modified_data.iloc[idx:end_idx, 0], min_limit, max_limit)

                # Record error period
                self.error_periods.append(
                    ErrorPeriod(
                        start_time=modified_data.index[idx],
                        end_time=modified_data.index[end_idx - 1],
                        error_type='noise',
                        original_values=original_values_for_log.values, # Log original values over the whole period
                        modified_values=modified_data.iloc[idx:end_idx].values,
                        parameters=period_parameters
                    )
                )
                
                self._mark_period_used(idx, end_idx)
                successful_injections += 1

        return modified_data

    def inject_baseline_shift_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject sudden baseline shift errors that permanently change the base level.
        Creates abrupt, permanent changes in the baseline level, similar to sensor recalibration
        or physical changes in measurement conditions.
        
        Calculates the total number of shifts based on data span (years) and configured
        count_per_year, then distributes them randomly throughout the entire dataset.
        """
        modified_data = data.copy()
        
        # Get baseline shift parameters from config
        shift_config = self.config['baseline shift']
        count_per_year = shift_config.get('count_per_year', 0)
        
        # If count_per_year is 0, don't inject any baseline shifts
        if count_per_year == 0:
            print("Baseline shift injection disabled (count_per_year = 0)")
            return modified_data
        
        # Calculate years in data and total number of shifts
        years_in_data = self._calculate_years_in_data(data.index)
        # Ensure at least 1 baseline shift if count_per_year > 0
        n_shifts = max(1, round(count_per_year * years_in_data))
        
        print(f"Attempting to inject {n_shifts} baseline shifts randomly across entire dataset...")
        
        # Try to inject shifts
        successful_shifts = 0
        attempts = 0
        max_attempts = n_shifts * 10
        
        while successful_shifts < n_shifts and attempts < max_attempts:
            # Select random index from entire dataset
            idx = np.random.randint(0, len(data) - 1)
            
            if self._is_period_available(idx, idx + 1):
                # Store original values
                original_values = modified_data.iloc[idx:idx+2].copy()
                
                # Determine shift direction and magnitude
                direction = np.random.choice([-1, 1], p=[shift_config['negative_positive_ratio'], 1-shift_config['negative_positive_ratio']])
                magnitude = direction * np.random.uniform(*shift_config['magnitude_range'])
                
                # Apply the shift (abrupt change)
                modified_data.iloc[idx+1:] += magnitude  # Everything after the shift point
                
                # Record error period
                self.error_periods.append(
                    ErrorPeriod(
                        start_time=modified_data.index[idx],
                        end_time=modified_data.index[idx + 1],
                        error_type='baseline shift',
                        original_values=original_values.values.flatten(),
                        modified_values=modified_data.iloc[idx:idx+2].values.flatten(),
                        parameters={
                            'magnitude': magnitude
                        }
                    )
                )
                
                # Mark shift point as used
                self._mark_period_used(idx, idx + 1)
                successful_shifts += 1
            
            attempts += 1
        
        return modified_data

    def inject_all_errors(self, data: pd.DataFrame, error_types: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject all types of errors into the data.
        
        For each error type, this method:
        1. Calculates the total number of errors based on years of data × count_per_year
        2. Distributes all errors randomly across the entire dataset
        3. Ensures errors don't overlap by tracking used periods
        
        This approach avoids creating regular patterns that a model might learn as normal,
        while still scaling the number of errors proportionally to the dataset size.
        """
        try:
            # Reset error periods and used indices at the START of each station/year
            self.error_periods = []
            self.used_indices = set()
            
            modified_data = data.copy()
            
            # Validate input data
            if data is None or data.empty:
                print("Warning: Empty or None data provided to inject_all_errors")
                return data, pd.DataFrame(index=data.index)
            
            # Default error types ordered by typical duration
            all_error_types = ['missing_data', 'offset', 'drift', 'flatline', 'spike', 'noise']
            
            # Filter error types based on count_per_year > 0 in config
            active_error_types = [
                et for et in all_error_types 
                if et in (error_types or all_error_types) 
                and self.config.get(et, {}).get('count_per_year', 0) > 0
            ]
            
            # Calculate years in data once
            years_in_data = self._calculate_years_in_data(data.index)
            print(f"Data spans approximately {years_in_data:.1f} years")
            print(f"Active error types: {active_error_types}")
            
            # Inject only active error types
            for error_type in active_error_types:
                try:
                    print(f"Injecting {error_type} errors...")
                    if error_type == 'spike':
                        modified_data = self.inject_spike_errors(modified_data)
                    elif error_type == 'flatline':
                        modified_data, _ = self.inject_flatline_errors(modified_data)
                    elif error_type == 'drift':
                        modified_data = self.inject_drift_errors(modified_data)
                    elif error_type == 'baseline shift':
                        modified_data = self.inject_baseline_shift_errors(modified_data)
                    elif error_type == 'offset':
                        modified_data, _ = self.inject_offset_errors(modified_data)
                    elif error_type == 'noise':
                        modified_data = self.inject_noise_errors(modified_data)
                    elif error_type == 'missing_data':
                        modified_data = self.inject_missing_data(modified_data)
                except Exception as e:
                    print(f"Error injecting {error_type}: {str(e)}")
                    continue
            
            # Update ground truth with error periods
            ground_truth = pd.DataFrame(index=data.index)
            ground_truth['error'] = False
            ground_truth['error_type'] = None
            
            for period in self.error_periods:
                mask = (ground_truth.index >= period.start_time) & (ground_truth.index <= period.end_time)
                ground_truth.loc[mask, 'error'] = True
                ground_truth.loc[mask, 'error_type'] = period.error_type
            
            print(f"Successfully injected {len(self.error_periods)} errors randomly across entire dataset")
            return modified_data, ground_truth
        
        except Exception as e:
            print(f"Error in inject_all_errors: {str(e)}")
            return data, pd.DataFrame(index=data.index)
